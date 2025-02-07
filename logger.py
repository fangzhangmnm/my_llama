import wandb
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import List,Callable,Union
from dataclasses import is_dataclass,asdict

class Logger:
    def __init__(self,folder,config=None,
                 use_wandb=False,wandb_project_name=None,wandb_run_name=None):
        # self.callbacks=[] # [(fn(iter),interval)]
        self.curves={} #{name:[(iter,value)]}
        self.config=self.json_handler(config)
        self.folder=folder
        self.use_wandb=use_wandb
        self.load()
        self.iteration=0
        if self.use_wandb:
            if hasattr(self,"wandb_run_id"):
                self.wandb_run=wandb.init(id=self.wandb_run_id,resume="allow",project=wandb_project_name,name=wandb_run_name,config=self.config)
            else:
                self.wandb_run=wandb.init(project=wandb_project_name,name=wandb_run_name,config=self.config)
        self.step_value_dict={}
    # def add_callback(self,fn:Callable[[int],None],interval:int=1):
    #     self.callbacks.append((fn,interval))
    def log(self,*args,**kwargs):
        if len(args)==1 and len(kwargs)==0: value_dict=args[0]
        elif len(args)==0 and len(kwargs)==1 and "value_dict" in kwargs: value_dict=kwargs["value_dict"]
        elif len(args)==0 and len(kwargs)>0: value_dict=kwargs
        elif len(args)>0 and len(args)%2==0 and len(kwargs)==0: value_dict=dict(zip(args[::2],args[1::2]))
        else: raise ValueError("use log({k:v}) or log(k=v) or log(k1,v1,k2,v2,...)")
        self.step_value_dict.update(value_dict)

    def step(self):
        self.step_value_dict["iteration"]=self.iteration
        for k,v in self.step_value_dict.items():
            self.curves.setdefault(k,[]).append((self.iteration,v))
        if self.use_wandb:
            wandb.log(self.step_value_dict)
        self.step_value_dict={}
        self.iteration+=1

    def save(self,i:int=None,path=None):
        try:
            if path is None: path=self.folder+"/log.json"
            os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
            with open(path,"w",encoding="utf-8") as f:
                data={
                    "config":self.config,
                    "curves":self.curves
                }
                if hasattr(self,"wandb_run"):
                    data["wandb_run_id"]=self.wandb_run.id
                json.dump(data,f,ensure_ascii=False,default=str)
        except Exception as e:
            print("Error in saving log",e)
    def load(self,path=None):
        if path is None: path=self.folder+"/log.json"
        if not os.path.exists(path): return
        try:
            with open(path,"r",encoding="utf-8") as f:
                data=json.load(f)
                self.curves=data.get("curves",{})
                if len(self.config)==0: # only load old one if no new config is set
                    self.config=data.get("config",{})
                if "wandb_run_id" in data:
                    self.wandb_run_id=data["wandb_run_id"]
        except Exception as e:
            print("Error in loading log",e)
    def reset_iteration(self,iteration:int):
        self.iteration=iteration
        for k in list(self.curves.keys()):
            self.curves[k]=[v for v in self.curves[k] if v[0]<iteration]
    @staticmethod
    def json_handler(o):
        if isinstance(o,dict):return {k:Logger.json_handler(v) for k,v in o.items()}
        elif isinstance(o,(list,tuple)):return [Logger.json_handler(v) for v in o]
        elif is_dataclass(o): return {k:Logger.json_handler(v) for k,v in asdict(o).items()}
        elif isinstance(o, (np.integer,int)): return int(o)
        elif isinstance(o, (np.floating,float)): return float(o)
        elif isinstance(o, np.ndarray): return o.tolist()
        elif isinstance(o, (np.bool_,bool)): return bool(o)
        else: return str(o)

class LogPlotter:
    @staticmethod
    def format_number(x):
        if abs(x)<1: return f"{x:.2g}"
        if abs(x)<1e3: return f"{x:.1f}"
        if abs(x)<1e6: return f"{x/1e3:.1f}K"
        if abs(x)<1e9: return f"{x/1e6:.1f}M"
        if abs(x)<1e12: return f"{x/1e9:.1f}G"
        return f"{x/1e12:.1f}T"
    @staticmethod
    def calculate_ylim(x):
        # remove nan, inf
        x=x[np.isfinite(x)]
        if len(x)==0: return [-1,1]
        lower=min(np.percentile(x,10),0)
        upper=max(np.percentile(x,90),0)
        return [lower-(upper-lower)*0.3,upper+(upper-lower)*0.3]
    def __init__(self,logger:Logger,compare_folders:Union[str,List[str]]=None):
        self.logger=logger
        self.widget=self.detect_ipython()
        if compare_folders is not None:
            if isinstance(compare_folders,str): compare_folders=[compare_folders]
            # self.compare_curves_sets=[Logger(f).curves for f in compare_folders]
            self.compare_curves_sets={os.path.basename(f):Logger(f).curves for f in compare_folders}
        else:
            self.compare_curves_sets={}
    def __call__(self,iteration:int):
        try:
            plt.rcParams.update({'font.size': 8})
            keys=self.logger.curves.keys()
            n_cols=3
            n_rows=(len(keys)+n_cols-1)//n_cols
            fig=plt.figure(figsize=(10,1.5*n_rows))
            for i,key in enumerate(keys):
                ax=plt.subplot(n_rows,n_cols,i+1)
                xy=np.array(self.logger.curves[key])
                plt.plot(xy[:,0],xy[:,1],label=key)
                for name,compare_curves in self.compare_curves_sets.items():
                    if key in compare_curves:
                        xy_cmp=np.array(compare_curves[key])
                        # plot behind
                        plt.plot(xy_cmp[:,0],xy_cmp[:,1],linestyle='--',alpha=0.3,label=name,zorder=-1)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: LogPlotter.format_number(x)))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune=None, integer=True))
                # add grid 
                ax.grid(True) 
                plt.title(key)
                ax.xaxis.set_ticks_position('top');ax.xaxis.set_label_position('top')
                if i>=n_cols: 
                    ax.set_xticklabels([])
                else: 
                    plt.xlabel("iteration")
                ax.set_ylim(self.calculate_ylim(xy[:,1]))
                ax.set_xlim([-1,iteration+1])
                if i==0 and self.compare_curves_sets is not None:
                    plt.legend()
            plt.suptitle(f"iteration {iteration}")
            plt.tight_layout()
            if not os.path.exists(os.path.abspath(self.logger.folder)):
                os.makedirs(os.path.abspath(self.logger.folder))
            plt.savefig(self.logger.folder+"/log.png")
            if self.widget:
                self.widget_stuff()
            plt.close(fig)
        except Exception as e:
            print("Error in plotting",e)
    def detect_ipython(self):
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except:
            return False
    def widget_stuff(self):
        import ipywidgets
        from IPython.display import display
        if not hasattr(self,"ctx"):
            self.ctx=ipywidgets.Output()
            display(self.ctx)
        interrupt=False
        with self.ctx:
            try:
                self.ctx.clear_output(wait=True)
                plt.show()
            except KeyboardInterrupt: interrupt=True
        if interrupt: raise KeyboardInterrupt()
