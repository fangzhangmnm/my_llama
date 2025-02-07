from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import zhconv
import random

import jsonlines,os,json

def cleanup_text(text):
    # text=normalize_unicode('NFKC',text) # don't help
    lines=text.split('\n')
    text=""
    for line in lines:
        line=zhconv.convert(line, 'zh-hans')
        line=line.replace('\r','')
        text+=line+'\n'
    return text

def iter_jsonls(input_jsonl_paths,drop_prob=None):
    for input_jsonl_path in input_jsonl_paths:
        with jsonlines.open(input_jsonl_path) as reader:
            for obj in reader:
                if drop_prob is not None and random.random()<drop_prob:
                    continue
                yield cleanup_text(obj['text'])


special_tokens = [
    "<unk>", 
    "<s>", "</s>",
    "<think>", "</think>",
    "<system_call>", "</system_call>",
    "<system>", "</system>",
]

vocab_size=6400


def train_tokenizer(dataiter, output_dir):
    random.seed(42)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer=trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()    
    )

    tokenizer.train_from_iterator(dataiter, trainer=trainer)

    tokenizer.decoder=decoders.ByteLevel()
    
    for idx, token in enumerate(special_tokens):
        assert tokenizer.token_to_id(token) == idx

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    tokenizer.model.save(output_dir)

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            str(idx): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            } for idx, token in enumerate(special_tokens)
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, "vocab_human_readable.txt"), "w") as f:
        for idx in range(len(tokenizer.get_vocab())):
            decoded=tokenizer.decode([idx])
            escaped=repr(decoded)
            escaped='"'+escaped[1:-1]+'"'
            f.write(f"{idx} {escaped}\n")

    print("Tokenizer saved to", output_dir)

def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages=[
        {"role": "system", "content": "你是一个聊天机器人。"},
        {"role": "user", "content": '今天天气怎么样？'},
        {"role": "assistant", "content": '<think>我需要通过天气API查询一下。</think><system_call>get_weather()</system_call><system>Sunny, 20 degree</system>今天天气不错。'},
    ]
    new_prompt=tokenizer.apply_chat_template(messages,tokenize=False)
    print(new_prompt)
    
    actual_vocab_size=len(tokenizer)
    print("Actual vocab size:", actual_vocab_size)

    model_inputs=tokenizer(new_prompt)
    print("encoded length:", len(model_inputs['input_ids']))

    input_ids=model_inputs['input_ids']
    response=tokenizer.decode(input_ids)

    for token in input_ids:
        print(token,tokenizer.decode([token]))

    print('response is the same as input:', response==new_prompt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # input a glob
    parser.add_argument("input_jsonl_paths", nargs="+")
    parser.add_argument("output_dir")
    parser.add_argument("--drop_prob", type=float, default=None)

    args = parser.parse_args()

    if len(args.input_jsonl_paths)>0:
        dataiter=iter_jsonls(args.input_jsonl_paths,drop_prob=args.drop_prob)
        train_tokenizer(dataiter, args.output_dir)
    eval_tokenizer(args.output_dir)


    
