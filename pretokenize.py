import os
from glob import glob
import zhconv
# from unicodedata import normalize as normalize_unicode
import jsonlines
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging

def get_tokenizer(tokenizer_path="./tokenizer"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)

def get_text_from_json(obj):
    guesses=["story_zh","text_zh","story","text","content","unstructured_text"]
    for guess in guesses:
        if guess in obj:
            return obj[guess]
    return max(obj.values(), key=len)

def cleanup_text(text):
    # text=normalize_unicode('NFKC',text) # don't help
    lines=text.split('\n')
    text=""
    for line in lines:
        line=zhconv.convert(line, 'zh-hans')
        line=line.replace('\r','')
        text+=line+'\n'
    return text

def iter_jsonlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as reader:
        for obj in jsonlines.Reader(reader):
            yield get_text_from_json(obj)


def iter_markdown(file_path):
    with open(file_path,encoding='utf-8') as reader:
        doclines=[]
        for line in reader:
            if line.startswith('#') and line.lstrip('#').startswith(' '):
                if len(doclines)>1: # skip title only docs
                    yield '\n'.join(doclines)
                doclines=[]
                line=line.lstrip('#').lstrip(' ')
            doclines.append(line)
        if len(doclines)>1: # skip title only docs
            yield '\n'.join(doclines)
            

def get_shard_name(shard_path):
    return '.'.join(os.path.basename(shard_path).split('.')[:-1])

def process_shard(args):
    try:
        shard_id, shard_path, tokenized_data_dir, prefix = args
        shard_name=get_shard_name(shard_path)
        output_path=os.path.join(tokenized_data_dir,shard_name+'.bin')
        index_path = os.path.join(tokenized_data_dir, shard_name + '.index')

        tokenizer=get_tokenizer()
        bos_id=tokenizer.bos_token_id
        eos_id=tokenizer.eos_token_id
        # bos_id=tokenizer.token_to_id('<s>')
        # eos_id=tokenizer.token_to_id('</s>')

        if shard_path.endswith('.jsonl'):
            texts=list(iter_jsonlines(shard_path))
        elif shard_path.endswith('.md') or shard_path.endswith('.txt'):
            texts=list(iter_markdown(shard_path))

        print("Start processing",shard_name)
        print("Cleaning up texts",shard_name)

        for i in range(len(texts)):
            texts[i]=cleanup_text(texts[i])

        print("Tokenizing texts",shard_name)

        for i in range(len(texts)):
            text=texts[i]
            if prefix is not None and len(prefix)>0:
                text=prefix+text
            tokens=tokenizer.encode(text)
            # tokens=tokenizer.encode(text).ids
            # remove possible heading bos and trailing eos
            if tokens[0]==bos_id: tokens=tokens[1:]
            if tokens[-1]==eos_id: tokens=tokens[:-1]
            # now add heading bos
            tokens=[bos_id]+tokens
            texts[i]=np.array(tokens,dtype=np.uint16)
        all_tokens=np.concatenate(texts)
        with open(output_path,'wb') as f:
            f.write(all_tokens.tobytes())
        lengths=np.array([0]+[len(text) for text in texts],dtype=np.uint32)
        offsets=np.cumsum(lengths,dtype=np.uint64)[:-1]
        with open(index_path,'wb') as f:
            f.write(offsets.tobytes())
        token_count=len(all_tokens)
        avg_seq_len=token_count//len(texts)
        print(f'Saved {output_path}, average sequence length: {avg_seq_len} , total tokens: {token_count}, number of texts: {len(texts)}')
    except Exception as e:
        print(f"Failed to process {shard_path}",shard_name,e)
        raise e

    # avg_seq_len=len(all_tokens)//len(texts)
    # print(f'Saved {output_path}, average sequence length: {avg_seq_len} , total tokens: {len(all_tokens)}, number of texts: {len(texts)}')

def pretokenize(input_data_dir,tokenized_data_dir,overwrite=False,max_workers=1,prefix=""):
    all_exts=['*.jsonl','*.md','*.txt']
    all_filenames=[]
    for ext in all_exts:
        all_filenames+=glob(input_data_dir+'/'+ext)
    if not overwrite:
        all_filenames=[f for f in all_filenames if not os.path.exists(os.path.join(tokenized_data_dir,get_shard_name(f)+'.bin'))]
    if max_workers>1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_shard,[(i,shard,tokenized_data_dir,prefix) for i,shard in enumerate(all_filenames)])
    else:
        for i,shard in enumerate(all_filenames):
            print(f"Processing {shard}")
            process_shard((None,shard,tokenized_data_dir))
    if len(all_filenames)>0:
        print(f"Finished processing {input_data_dir} to {tokenized_data_dir}")



# input_train_data_dir="./train_data"
# tokenized_train_data_dir="./tokenized_train_data"
# input_val_data_dir="./val_data"
# tokenized_val_data_dir="./tokenized_val_data"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',help='input data directory')
    parser.add_argument('tokenized_dir',help='tokenized data directory')
    parser.add_argument('-o','--overwrite',action='store_true',help='overwrite existing tokenized data')
    parser.add_argument('--max_workers',type=int,default=0)
    parser.add_argument('--prefix',type=str,default="")
    args = parser.parse_args()
    if args.max_workers==0:
        args.max_workers=os.cpu_count()
    args.prefix=args.prefix.replace('\\n','\n')
    os.makedirs(args.tokenized_dir, exist_ok=True)
    # if contains jsonl, md, txt files, pretokenize them
    pretokenize(args.input_dir,args.tokenized_dir,overwrite=args.overwrite,max_workers=args.max_workers,prefix=args.prefix)
    for folder in os.listdir(args.input_dir):
        if os.path.isdir(os.path.join(args.input_dir,folder)):
            input_data_dir=os.path.join(args.input_dir,folder)
            tokenized_data_dir=os.path.join(args.tokenized_dir,folder)
            os.makedirs(tokenized_data_dir,exist_ok=True)
            pretokenize(input_data_dir,tokenized_data_dir,overwrite=args.overwrite,max_workers=args.max_workers)

# python pretokenize.py /mnt/d/AIDATA/DATASETS/input/ ./data/version/subfolder/ --prefix "set_name\n"
