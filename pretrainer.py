from dataclasses import dataclass, asdict
from logger import Logger
from model import TransformerModel
import torch
from torch.utils.data import DataLoader, Dataset
import shutil
import math,time,os,inspect
from typing import List,Callable,Union,Literal
import signal
from trainer_base import BaseTrainer
from trainer_mods import stablemax_cross_entropy, with_orthogonal_gradient

@dataclass
class LLMPreTrainConfig:
    ## batch and minibatch
    tokens_per_iter:int # determines number of batches before gradient update
    max_seq_len:int
    batch_size:int # depends on GPU RAM
    ## lr and optimizer
    optimizer_type:Literal['AdamW','AdamW_ortho']
    loss_fn:Literal['cross_entropy','stablemax_cross_entropy']
    stable_lr:float
    min_lr:float
    weight_decay:float
    beta1:float
    beta2:float
    grad_norm_clip:float # 0 to disable
    ## lr scheduler
    warmup_iters:int
    stable_iters:int
    decay_iters:int
    finetuning_iters:int
    ## path
    checkpoint_dir:str
    ## intervals
    log_interval:int
    eval_interval:int
    backup_checkpoint_interval:int
    checkpoint_interval:int
    # dataset shuffling
    iter_per_shard:int
    max_opened_shards:int
    ## hardware
    device:str # 'cuda' or 'cpu'
    dtype:str # float32|bfloat16|float16
    compile:Union[bool,str] # bool or 'max-autotune' 'reduce-overhead'
    fp8_training:bool
    num_workers:int # for DataLoader
    use_gradient_checkpoint:bool

    def __post_init__(self):
        if not self.tokens_per_iter/(self.batch_size*self.max_seq_len)>=1:
            raise ValueError("batch_size too large!")
        self.gradient_accumulation_steps=self.tokens_per_iter//(self.batch_size*self.max_seq_len)
        if self.tokens_per_iter!=self.batch_size*self.max_seq_len*self.gradient_accumulation_steps:
            self.tokens_per_iter=self.batch_size*self.max_seq_len*self.gradient_accumulation_steps
            print(f"Adjusted tokens_per_iteration to {self.tokens_per_iter} to match batch_size*max_seq_len")
        self.max_iters=self.warmup_iters+self.stable_iters+self.decay_iters+self.finetuning_iters


class LLMPreTrainer(BaseTrainer):
    cf:LLMPreTrainConfig
    def __init__(self,
                model:TransformerModel,
                dataset:Dataset,
                config:LLMPreTrainConfig,
                logger:Logger
    ):
        super().__init__(config=config,logger=logger)
        self.model = model
        self.dataset = dataset
        self.create_optimizer()
        self.data_iterator = iter(DataLoader(
            self.dataset,
            batch_size=self.cf.batch_size,
            pin_memory=True,
            num_workers=self.cf.num_workers,
            worker_init_fn=self.get_worker_init_fn(),
            shuffle=False
        ))
        self.model.to(self.cf.device)
        if self.cf.fp8_training:
            self.model=self.convert_float8(self.model)
        if isinstance(self.cf.compile,str) or self.cf.compile:
            compile_mode=self.cf.compile if isinstance(self.cf.compile,str) else None
            print("Compiling model, takes ~1min")
            self.model=torch.compile(self.model,mode=compile_mode,fullgraph=True)
            print("Model compiled")
        dtype_obj={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}
        self.ctx=torch.amp.autocast(device_type=self.cf.device,dtype=dtype_obj[self.cf.dtype])

        print("Config:")
        print(asdict(self.cf))
        print("Model Args:")
        print(asdict(self.model.args))
        print("Model:")
        print(self.model)

    def convert_float8(self,model):
        # more beta features to explore, like delayed scaling. see https://github.com/pytorch/ao/blob/main/torchao/float8/README.md
        from torchao.float8 import convert_to_float8_training
        model = convert_to_float8_training(model, config=None, module_filter_fn=None)
        # don't need to worry about weight tying. it is being automatically handled by torchao
        return model
    
    def _train_step(self)->bool:
        if self.iteration>=self.cf.max_iters:
            return True
        lr=self.get_lr()
        iter_loss=0
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr
        self.model.train()
        for micro_step in range(self.cf.gradient_accumulation_steps):
            with self.ctx:
                X=next(self.data_iterator)
                X,Y=X[...,:-1],X[...,1:]
                X=X.to(self.cf.device,non_blocking=True,dtype=torch.long)
                Y=Y.to(self.cf.device,non_blocking=True,dtype=torch.long)
                logits=self.model(
                    X,
                    use_gradient_checkpoint=self.cf.use_gradient_checkpoint)
                if self.cf.loss_fn=='cross_entropy':
                    loss=torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),Y.view(-1),ignore_index=-1,reduction='none')
                    loss=(loss).mean()/self.cf.gradient_accumulation_steps
                elif self.cf.loss_fn=='stablemax_cross_entropy':
                    loss=stablemax_cross_entropy(logits.view(-1,logits.size(-1)),Y.view(-1),reduction='none')
                    loss=(loss.view(-1)).mean()/self.cf.gradient_accumulation_steps
                else:
                    raise ValueError(f"Unknown loss function {self.cf.loss_fn}")
            self.scaler.scale(loss).backward()
            iter_loss+=loss.item()
        if self.cf.grad_norm_clip>0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cf.grad_norm_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True) # save memory immediately

        self.logger.log({
            "tokens":(self.iteration+1)*self.cf.tokens_per_iter,
            "loss":iter_loss,
            "lr":lr,
        })
        self.set_iter_info(
            loss=iter_loss,
            tokens=(self.iteration+1)*self.cf.tokens_per_iter
            )
        
            

    def create_optimizer(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.cf.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters) and (self.cf.device == 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        if self.cf.optimizer_type=='AdamW':
            self.optimizer = torch.optim.AdamW(
                optim_groups, 
                lr=self.cf.stable_lr, 
                betas=(self.cf.beta1,self.cf.beta2), 
                **extra_args)
        elif self.cf.optimizer_type=='AdamW_ortho':
            self.optimizer = with_orthogonal_gradient(torch.optim.AdamW)(
                optim_groups,
                skip_orthogonal_1d=True,
                skip_orthogonal_param_types=['bias','norm'],
                lr=self.cf.stable_lr,
                betas=(self.cf.beta1,self.cf.beta2),
                weight_decay=self.cf.weight_decay,
                **extra_args)
        # AdamWFp8_ortho might not be very useful, will implement in the future when library gets stable and receipt are rolled out
        else:
            raise ValueError(f"Unknown optimizer type {self.cf.optimizer_type}")
        self.scaler=torch.amp.GradScaler(device=self.cf.device,
                                         enabled=self.cf.dtype == 'float16') #bf16 and float32 are safe

    def get_lr(self):
        if self.iteration<self.cf.warmup_iters:
            return self.cf.stable_lr*self.iteration/self.cf.warmup_iters
        elif self.iteration<self.cf.warmup_iters+self.cf.stable_iters:
            return self.cf.stable_lr
        elif self.iteration<self.cf.warmup_iters+self.cf.stable_iters+self.cf.decay_iters:
            decay_it=self.iteration-self.cf.warmup_iters-self.cf.stable_iters
            return decay_function(decay_it/self.cf.decay_iters,self.cf.stable_lr,self.cf.min_lr)
        else:
            return self.cf.min_lr

    def _serialize_checkpoint(self):
        return {
            "iteration":self.iteration,
            "model":self.model.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "scaler":self.scaler.state_dict(),
            "model_args":asdict(self.model.args),
            "training_config":asdict(self.cf),
        }
    def _deserialize_checkpoint(self, checkpoint):
        self.iteration=checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.logger.reset_iteration(self.iteration)


def load_model_from_checkpoint(ckpt_path):
    from model import TransformerModel, TransformerModelArgs
    checkpoint = torch.load(ckpt_path,weights_only=True)
    model_args = TransformerModelArgs(**checkpoint['model_args'])
    model = TransformerModel(model_args)
    state_dict=checkpoint['model']
    state_dict={k[10:] if k.startswith('_orig_mod.') else k: v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def decay_function(x,ymax,ymin):
    # exponentially lerp between ymax and ymin
    log_ymax,log_ymin=math.log(ymax),math.log(ymin)
    return math.exp(log_ymax*(1-x)+log_ymin*x)
