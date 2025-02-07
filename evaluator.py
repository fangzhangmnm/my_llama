from dataset import PretokDataset
from model import TransformerModel
from logger import Logger

import os, glob, time, random, torch
from torch.utils.data import DataLoader

class TrainLossEvaluator:
    def __init__(
            self,
            eval_dataset_dir: str,
            model: TransformerModel,
            logger: Logger,
            max_seq_len: int = 1024,
            n_eval_samples: int=5,
    ):
        self.data_iterators={}
        for filename in glob.glob(eval_dataset_dir+'/*.bin'):
            dataset=PretokDataset(filename,max_seq_len=max_seq_len+1)
            eval_name=os.path.basename(filename).split('.')[0]
            self.data_iterators[eval_name]=iter(DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=True,
            ))
        self.model=model
        self.logger=logger
        self.n_eval_samples=n_eval_samples
    def __call__(self,iters:int):
        self.model.eval()
        device=next(self.model.parameters()).device
        with torch.no_grad():
            # pick at most n_eval_samples from avaliable eval_name
            eval_names=random.sample(list(self.data_iterators.keys()),min(len(self.data_iterators),self.n_eval_samples))
            eval_result={}
            for eval_name in eval_names:
                data_iterator=self.data_iterators[eval_name]
                X=next(data_iterator)
                X,Y=X[...,:-1],X[...,1:]
                X=X.to(device,non_blocking=True,dtype=torch.long)
                Y=Y.to(device,non_blocking=True,dtype=torch.long)
                logits=self.model(X)
                loss=torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),Y.view(-1),ignore_index=-1,reduction='none')
                loss=(loss).mean()
                eval_result["loss/"+eval_name]=loss.item()
            self.logger.log(eval_result)
        self.model.train()
                                


        