from torch.utils.data import DataLoader, Dataset, IterableDataset
import yaml,os
import glob
import numpy as np
import random

def open_shard(shard_path,index_path):
    shard=np.memmap(shard_path, dtype=np.uint16, mode='r')
    starts = np.fromfile(index_path, dtype=np.uint64)
    ends=np.concatenate([starts[1:],np.array([shard.size],dtype=np.uint64)])
    return shard,starts,ends

def iter_shard_random(shard,starts,ends,seq_len):
    lengths=ends-starts
    probs=lengths/lengths.sum()
    slices=[]
    total_len=0
    while True:
        while total_len<seq_len:
            i=np.random.choice(len(starts),p=probs)
            s=shard[starts[i]:ends[i]]
            slices.append(s)
            total_len+=len(s)
        s=np.concatenate(slices)
        yield np.array(s[:seq_len])
        slices=[s[seq_len:]]
        total_len=len(slices[0])

def calculate_probs(dataset_dir, stage_weights, calculate_size_fn):
    stage_folder_probs={} # stage => ([folder],[probs])
    for stage in stage_weights:
        weights={k:v for k,v in stage_weights[stage].items() if v>0}
        folders=list(weights.keys())
        probs=np.array(list(weights.values()),dtype=np.float32)
        probs/=probs.sum()
        stage_folder_probs[stage]=(folders,probs)
    all_folders=list(set([k for weights in stage_weights.values() for k in weights.keys() ]))
    item_counts={}
    folder_files={}
    all_files=[]
    for folder in all_folders:
        for file_path in glob.glob(os.path.join(dataset_dir,folder,'*.bin')):
            folder_files.setdefault(folder,[]).append(file_path)
            all_files.append(file_path)
            print("scanning dataset",file_path)
            item_counts[file_path]=calculate_size_fn(file_path)
    folder_file_probs={}
    for folder in all_folders:
        files=folder_files[folder]
        files=[f for f in files if item_counts[f]>0]
        probs=np.array([item_counts[file_path] for file_path in files],dtype=np.float32)
        probs/=probs.sum()
        folder_file_probs[folder]=(files,probs)
    return all_files,stage_folder_probs,folder_file_probs

class PretokDataset(IterableDataset):
    def __init__(self, file_path, max_seq_len=1024):
        self.file_path=file_path
        self.max_seq_len=max_seq_len
    def __iter__(self):
        index_path=self.file_path.replace('.bin','.index')
        m,s,e=open_shard(self.file_path,index_path)
        for X in iter_shard_random(m,s,e,self.max_seq_len):
            yield X

class PretrainDataset(IterableDataset):
    def __init__(self, dataset_dir, seq_len=1024, iter_per_shard=1024, max_opened_shards=16):
        self.dataset_dir = dataset_dir
        self.seq_len=seq_len
        self.stage = 'warmup'
        with open(os.path.join(dataset_dir,'weights.yaml'), 'r') as file:
            stage_weights = yaml.safe_load(file)
        get_size=lambda file_path: os.path.getsize(file_path)//2
        all_files,self.stage_folder_probs,self.folder_file_probs=calculate_probs(self.dataset_dir, stage_weights, get_size)
        self.iter_per_shard=iter_per_shard
        self.max_opened_shards=max_opened_shards
        self.shard_iterators=[]

    def choose_random_file(self):
        folders,probs=self.stage_folder_probs[self.stage]
        folder=np.random.choice(folders,p=probs)
        files,probs=self.folder_file_probs[folder]
        file_path=np.random.choice(files,p=probs)
        return file_path
    
    def __iter__(self):
        while True:
            while len(self.shard_iterators)<self.max_opened_shards:
                file_path=self.choose_random_file()
                shard_path=file_path
                index_path=file_path.replace('.bin','.index')
                shard,starts,ends=open_shard(shard_path,index_path)
                it=enumerate(iter_shard_random(shard,starts,ends,self.seq_len))
                self.shard_iterators.append(it)
            i=np.random.choice(len(self.shard_iterators))
            it=self.shard_iterators[i]
            count,X=next(it)
            if count>=self.iter_per_shard:
                self.shard_iterators.pop(i)
            yield X



if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('data_dir',type=str)
    parser.add_argument('--max_seq_len',type=int,default=1024)
    parser.add_argument('--iter_per_shard',type=int,default=1024)
    parser.add_argument('--max_opened_shards',type=int,default=16)
    parser.add_argument('--tokenizer',type=str,default='./tokenizer')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    dataset=PretrainDataset(
        args.data_dir,
        seq_len=args.max_seq_len,
        iter_per_shard=args.iter_per_shard,
        max_opened_shards=args.max_opened_shards
    )
    def get_tokenizer(tokenizer_path="./tokenizer"):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer=get_tokenizer(args.tokenizer)
    # benchmark 1000 samples
    if args.benchmark:
        import time
        t0=time.time()
        for i,(X,Y) in enumerate(dataset):
            if i==1000:
                break
        t1=time.time()
        print("1000 samples in",t1-t0,"seconds")
    else:
        for X,Y in dataset:
            if input("input q to quit:\n")=="q":
                exit()
            XY=np.concatenate([X, Y[-1:]])
            print(tokenizer.decode(XY))