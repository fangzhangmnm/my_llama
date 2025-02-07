from model import TransformerModel, TransformerModelArgs, TransformerBlock
import torch,os,json,typing,glob,shutil


def load_hf_model(directory):
    json_path = os.path.join(directory, "config.json")
    bin_path = os.path.join(directory, "pytorch_model.bin")
    if os.path.exists(bin_path):
        hf_state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
    else:
        from safetensors.torch import load_file
        # bin_path=os.path.join(directory, "model.safetensors")
        # hf_state_dict = load_file(bin_path, device='cpu')
        safetensor_paths = glob.glob(os.path.join(directory, "*.safetensors"))
        hf_state_dict = {}
        for safetensor_path in safetensor_paths:
            hf_state_dict.update(load_file(safetensor_path, device='cpu'))
    with open(json_path) as f: 
        hf_config = json.load(f)
    print("state dict",list(hf_state_dict.keys()))
    
    attn_bias='model.layers.0.self_attn.q_proj.bias' in hf_state_dict
    model_args = TransformerModelArgs(
        dim=hf_config['hidden_size'],
        n_layers=hf_config['num_hidden_layers'],
        n_heads=hf_config['num_attention_heads'],
        n_kv_heads=hf_config['num_key_value_heads'],
        vocab_size=hf_config['vocab_size'],
        hidden_dim=hf_config['intermediate_size'],
        norm_eps=hf_config['rms_norm_eps'],
        max_position_embeddings=hf_config['max_position_embeddings'],
        dropout=hf_config['attention_dropout'],
        embedding_weight_tying=hf_config['tie_word_embeddings'],
        rope_theta=hf_config['rope_theta'],
        attn_bias=attn_bias
    )
    print("model args",model_args)
    try:
        model = TransformerModel(model_args)
        model.embedding.weight.data = hf_state_dict['model.embed_tokens.weight']
        model.norm.weight.data = hf_state_dict['model.norm.weight']

        for layer_idx in range(model_args.n_layers):
            layer:TransformerBlock = model.layers[layer_idx]
            layer.attention_norm.weight.data = hf_state_dict[f'model.layers.{layer_idx}.input_layernorm.weight']
            layer.attention.wq.weight.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight']
            layer.attention.wk.weight.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight']
            layer.attention.wv.weight.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight']
            if attn_bias:
                layer.attention.wq.bias.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.bias']
                layer.attention.wk.bias.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.bias']
                layer.attention.wv.bias.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.bias']
            layer.attention.wo.weight.data = hf_state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight']
            layer.ffn_norm.weight.data = hf_state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight']
            layer.feed_forward.w1.weight.data = hf_state_dict[f'model.layers.{layer_idx}.mlp.gate_proj.weight']
            layer.feed_forward.w2.weight.data = hf_state_dict[f'model.layers.{layer_idx}.mlp.down_proj.weight']
            layer.feed_forward.w3.weight.data = hf_state_dict[f'model.layers.{layer_idx}.mlp.up_proj.weight']

        if 'lm_head.weight' not in hf_state_dict:
            model_args.embedding_weight_tying = True
        elif torch.equal(hf_state_dict['lm_head.weight'], hf_state_dict['model.embed_tokens.weight']):
            model_args.embedding_weight_tying = True
        else:
            model_args.embedding_weight_tying = False
        if model_args.embedding_weight_tying:
            model.output.weight = model.embedding.weight
        else:
            model.output.weight.data = hf_state_dict['lm_head.weight']
    except KeyError as e:
        print(f"Missing key in state dict: {e}")
        print("Available keys in state dict:", hf_state_dict.keys())
        raise e

    return model

def hf_export(model:TransformerModel, filepath, dtype=torch.float32):
    """ Generate the pytorch_model.bin state_dict and config.json for HuggingFace """
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None
    
    args=model.args
    hf_state_dict={}

    hf_state_dict['model.embed_tokens.weight']=model.embedding.weight.clone().to(dtype)
    hf_state_dict['model.norm.weight']=model.norm.weight.clone().to(dtype)

    for layer_idx in range(args.n_layers):
        layer:TransformerBlock = model.layers[layer_idx]
        hf_state_dict[f'model.layers.{layer_idx}.input_layernorm.weight']=layer.attention_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight']=layer.attention.wq.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight']=layer.attention.wk.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight']=layer.attention.wv.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight']=layer.attention.wo.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight']=layer.ffn_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.mlp.gate_proj.weight']=layer.feed_forward.w1.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.mlp.down_proj.weight']=layer.feed_forward.w2.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{layer_idx}.mlp.up_proj.weight']=layer.feed_forward.w3.weight.clone().to(dtype)

    # llama2.c usually uses tied weights -> reference the embed_tokens.weights instead
    if args.embedding_weight_tying:
        hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']
    else:
        hf_state_dict['lm_head.weight'] = model.output.weight.clone().to(dtype)
        

    config=LlamaConfig(
        hidden_size=args.dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        intermediate_size=args.hidden_dim,
        rms_norm_eps=args.norm_eps,
        max_position_embeddings=args.max_position_embeddings,
        attention_dropout=args.dropout,
        tie_word_embeddings=args.embedding_weight_tying,
        rope_theta=args.rope_theta,
        # Manual
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )

    os.makedirs(filepath, exist_ok=True)
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)
    

def load_checkpoint(checkpoint:str,weights_only=True) -> TransformerModel:
    checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=weights_only)
    print(checkpoint['model_args'])
    model_args=TransformerModelArgs(**checkpoint['model_args'])
    model=TransformerModel(model_args)
    state_dict={k[10:] if k.startswith('_orig_mod.') else k: v for k,v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    return model

def copy_tokenizer(tokenizer_dir:str, output_dir:str):
    os.makedirs(output_dir, exist_ok=True)
    for filename in glob.glob(os.path.join(tokenizer_dir, '*')):
        if os.path.isfile(filename):
            shutil.copy(filename, os.path.join(output_dir, os.path.basename(filename)))

import argparse
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=str)
    parser.add_argument('tokenizer_dir',type=str)
    parser.add_argument('output_dir',type=str)
    parser.add_argument('--dtype',type=str,default='float32',choices=['float32','bfloat16'])
    args=parser.parse_args()

    model=load_checkpoint(args.checkpoint)
    copy_tokenizer(args.tokenizer_dir, args.output_dir)
    hf_export(model,args.output_dir,dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[args.dtype])
    print(f"Exported model to {args.output_dir}")



