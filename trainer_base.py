import os,shutil,torch,time,signal
from dataclasses import dataclass,asdict
from logger import Logger
from abc import ABC,abstractmethod
from typing import Callable

class BaseTrainer(ABC):
    """
    Handles checkpointing, logging/plotting, training loop, exit signals, profiling

    Please implement `_serialize_checkpoint`, `_deserialize_checkpoint`, `_train_step`

    api:

    - `add_callback((function,interval,post_step))`
        - `post_step` means the callback is called after `logger.step()`, ideal for plotting
    """
    def __init__(self,config,logger:Logger):
        assert all(x in asdict(config) for x in [
            'checkpoint_dir','log_interval','backup_checkpoint_interval','checkpoint_interval'])
        self.cf=config
        self.logger=logger
        self.iteration=0
        self.__callbacks=[] # [(fn(iter),interval,post_step)]
        self._end_now=False
        self.__iter_info={}

    def add_callback(self,fn:Callable[[int],None],interval:int=1,post_step:bool=False):
        self.__callbacks.append((fn,interval,post_step))

    def set_iter_info(self,**kwargs):
        self.__iter_info.update(kwargs)

    @abstractmethod
    def _serialize_checkpoint(self)->dict:
        pass
    @abstractmethod
    def _deserialize_checkpoint(self,checkpoint:dict):
        pass



    @abstractmethod
    def _train_step(self)->bool:
        # return True to stop training
        pass

    def train(self,resume=True):
        if resume:
            ckpt_path=os.path.join(self.cf.checkpoint_dir,"latest.pt")
            if os.path.exists(ckpt_path):
                self.load_checkpoint(ckpt_path)
        self.logger.reset_iteration(self.iteration)
        print("Training started")
        t_start=time.time()
        local_iteration=0
        signal.signal(signal.SIGINT,lambda *args: self._handle_sigint(*args))
        finished=False
        while not finished:

            # call implemented method
            finished=self._train_step()

            t_end=time.time()
            t_iteration,t_start=t_end-t_start,t_end
            vram_usage=torch.cuda.max_memory_allocated() if self.cf.device=='cuda' else 0

            if self.iteration%self.cf.log_interval==0:
                self.logger.log({
                    "iter_time":t_iteration,
                    "vram":vram_usage,
                })
                print(f"iter:{self.iteration} iter_time:{t_iteration:.2f} vram:{vram_usage/1024**3:.2f}GB",end=" ")
                for k,v in self.__iter_info.items():
                    if isinstance(v,float):
                        print(f"{k}:{v:.4f}",end=" ")
                    elif isinstance(v,int):
                        print(f"{k}:{format_number(v)}",end=" ")
                    else:
                        print(f"{k}:{v}",end=" ")
                print()

            for fn,interval,post_step in self.__callbacks:
                if self.iteration%interval==0 and not post_step:
                    fn(self.iteration)

            self.iteration+=1
            local_iteration+=1
            self.logger.step()

            for fn,interval,post_step in self.__callbacks:
                if self.iteration%interval==0 and post_step:
                    fn(self.iteration)

            do_backup_ckpt=self.cf.backup_checkpoint_interval>0 and self.iteration%self.cf.backup_checkpoint_interval==0
            do_save_ckpt=self.cf.checkpoint_interval>0 and self.iteration%self.cf.checkpoint_interval==0
            do_save_ckpt=do_save_ckpt or do_backup_ckpt
            if do_save_ckpt:
                self.save_checkpoint(backup=do_backup_ckpt)
                self.logger.save(self.iteration)
            if self._end_now: finished=True
        # end while
        if local_iteration>0:
            self.save_checkpoint(backup=False)
            self.logger.save(self.iteration)


    def save_checkpoint(self,backup=False):
        os.makedirs(self.cf.checkpoint_dir,exist_ok=True)
        bkup_name=f'ckpt_{self.iteration}.pt'
        ckpt_path = os.path.join(self.cf.checkpoint_dir,'latest.pt')
        bkup_path = os.path.join(self.cf.checkpoint_dir,bkup_name)
        checkpoint=self._serialize_checkpoint()
        torch.save(checkpoint,ckpt_path)
        if backup:
            shutil.copy(ckpt_path,bkup_path)
            print(f"Checkpoint saved at {ckpt_path} and backup at {bkup_path}")
        else:
            print(f"Checkpoint saved at {ckpt_path}")

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path,weights_only=True)
        self._deserialize_checkpoint(checkpoint)
        self.logger.reset_iteration(self.iteration)                                                                        
        print(f"Checkpoint loaded from {ckpt_path}")


    def _handle_sigint(self,signal,frame):
        if self._end_now:
            print("Ending immediately, current iteration will not be saved")
            import psutil
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                child.terminate()
            exit(0)
        else:
            print("Training interrupted")
            print("Will save checkpoint after finishing current iteration")
            print("Press Ctrl+C again to end immediately")
            self._end_now=True

    def get_worker_init_fn(self):
        def worker_init_fn(worker_id):
            signal.signal(signal.SIGINT,signal.SIG_IGN)
        return worker_init_fn
    

def format_number(val):
    # use K M T Y
    if val<1e3: return str(val)
    elif val<1e6: return f'{val/1e3:.1f}K'
    elif val<1e9: return f'{val/1e6:.1f}M'
    elif val<1e12: return f'{val/1e9:.1f}B'
    else: return f'{val/1e12:.1f}T'