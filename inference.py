if __name__ == '__main__':
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=str)
    parser.add_argument('--hf',action='store_true')
    parser.add_argument('--tokenizer',type=str,default='./tokenizer')
    parser.add_argument('--device',type=str,default='auto',choices=['cpu','cuda','auto'])
    parser.add_argument('--max_input_length',type=int,default=1024)
    parser.add_argument('--max_output_length',type=int,default=1024)
    parser.add_argument('-t','--temperature',type=float,default=0.5)
    # parser.add_argument('--mode',type=str,choices=['completion','chat'],default='chat')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('-c','--completion',action='store_const',const='completion',dest='mode')
    group.add_argument('--chat',action='store_const',const='chat',dest='mode')
    group.add_argument('--compile',type=bool,default=False)
    group.set_defaults(mode='chat')
    args=parser.parse_args()

    from model import TransformerModel,TransformerModelArgs
    import os,dataclasses,sys
    import torch
    import random,time
    # def get_tokenizer(tokenizer_path="./tokenizer"):
    #     import os
    #     from tokenizers import SentencePieceBPETokenizer
    #     return SentencePieceBPETokenizer(
    #         # vocab=tokenizer_path+"tokenizer-vocab.json",
    #         vocab=os.path.join(tokenizer_path, 'vocab.json'),
    #         merges=os.path.join(tokenizer_path, 'merges.txt'),
    #         add_prefix_space=False, # Chinese text doesn't need prefix space
    #     )
    def get_tokenizer(tokenizer_path="./tokenizer"):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path)
    
    random.seed(time.time())

    if args.hf:
        from export import load_hf_model
        model=load_hf_model(args.model)
    else:
        ckpt_path=args.model
        if args.device=='auto':
            args.device='cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint=torch.load(ckpt_path,map_location=args.device,weights_only=True)
        model_args=TransformerModelArgs(**checkpoint['model_args'])
        model=TransformerModel(model_args)
        state_dict=checkpoint['model']
        state_dict={k[10:] if k.startswith('_orig_mod.') else k: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)


    if args.compile:
        model=torch.compile(model)
    tokenizer=get_tokenizer(args.tokenizer)
    end_tokens=['<s>','\n问：','\n问:','\n问题:']

    print(f'模型"{args.model}"加载成功')

    if args.mode=='completion':
        print('欢迎使用AI文本生成，输入exit或quit或ctrl+z退出，输入clear或ctrl+c清空记录，输入回车继续对话生成')
        print('请输入文本开头：',end='')
        dialogue=""
    elif args.mode=='chat':
        print('欢迎使用AI聊天机器人，输入exit或quit或ctrl+z退出，输入clear或ctrl+c清空对话记录，输入回车继续对话')
        dialogue=""
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    def get_input(prompt):
        try:
            return input(prompt)
        except KeyboardInterrupt:
            print('clear')
            return 'clear'
        except EOFError:
            print('exit')
            return 'exit'

    new_token=None
    while True:
        try:
            if args.mode=='chat':
                user_input=get_input('问：')
            else:
                user_input=get_input('…')
            if user_input=='exit' or user_input=='quit':
                break
            elif user_input=='clear':
                dialogue=""
                print('对话记录已清空')
                continue
            elif len(user_input)>0:
                if args.mode=='chat':
                    # dialogue+='<|im_start|>user\n'+user_input+'\n<|im_end|>\n<|im_start|>assistant\n'
                    dialogue+='\n问：'+user_input+'\n答：'
                    print("答：",end='',flush=True)
                else:
                    dialogue+=user_input
            dialogue=dialogue[-args.max_input_length:]
            # ids=tokenizer.encode(dialogue).ids
            ids=tokenizer.encode(dialogue)
            output_ids=[]
            output_ids_buffer=[]
            for step,new_id in enumerate(model.stream_generate(
                    torch.tensor(ids).unsqueeze(0),
                    max_new_tokens=args.max_output_length,
                    temperature=args.temperature)):
                new_id=new_id.item()
                new_token=tokenizer.decode([new_id])
                output_ids.append(new_id)
                output_ids_buffer.append(new_id)
                output_word=tokenizer.decode(output_ids_buffer)
                if not output_word.endswith('�'):
                    print(output_word,end='',flush=True)
                    dialogue+=output_word
                    output_ids_buffer.clear()

                should_break=False
                for end_token in end_tokens:
                    if dialogue.endswith(end_token):
                        should_break=True
                        break
                if should_break:
                    break

                # if step>0 and new_token in end_tokens:
                #     # dialogue+="<|im_end|>\n"
                #     dialogue+=new_token
                #     print("",flush=True)
                #     # print(repr(dialogue),flush=True)
                #     break
                # elif dialogue.endswith("<|im_end|>"):
                #     dialogue+='\n'
                #     print("",flush=True)
                #     break
        except KeyboardInterrupt:
            dialogue=""
            print('对话记录已清空')
