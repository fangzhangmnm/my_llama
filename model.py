from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


@dataclass
class TransformerModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    hidden_dim: int = None
    norm_eps: float = 1e-5
    max_position_embeddings: int = 1024 # for RoPE
    dropout: float = 0.0
    embedding_weight_tying: bool = True
    rope_theta: float = 10000.0
    attn_bias: bool = False
    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = math.ceil(8/3 * self.dim/ 256)  * 256 


class TransformerModel(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.dropout = nn.Dropout(args.dropout)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.embedding_weight_tying:
            self.embedding.weight = self.output.weight
        self.init_weights(args)

    def init_weights(self, args: TransformerModelArgs):
        for name,module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('w3') or name.endswith('wo'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))
                else:
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)


    def forward(self, 
                ids: torch.Tensor, 
                kv_caches: Optional[List[KVCache]] = None, 
                attn_mask: Optional[torch.Tensor] = None,
                use_gradient_checkpoint: bool = False
                )->torch.Tensor:
        '''
        input: (batch_size, sequence_length) token indices

        output: (batch_size, sequence_length, vocab_size) probabilities of the next token
        '''
        x = self.embedding(ids)
        x = self.dropout(x)
        for layer_idx, layer in enumerate(self.layers):
            kv_cache = kv_caches[layer_idx] if kv_caches is not None else None
            if use_gradient_checkpoint:
                x = checkpoint(layer, x, kv_cache, attn_mask)
            else:
                x = layer(x, kv_cache=kv_cache, attn_mask=attn_mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    @torch.inference_mode()
    def stream_generate(self, ids, max_new_tokens, temperature=1.0, save_logits=False):
        '''
        input: (batch_size, sequence_length) token indices
        output: yield (batch_size, 1) next token indices
        '''
        kv_caches = [KVCache() for _ in range(self.args.n_layers)]
        # kv_caches = None
        for step in range(max_new_tokens):
            ids = ids[:, -self.args.max_position_embeddings:]
            if step>0:
                ids = ids[:, -1:]
            logits = self(ids, kv_caches=kv_caches) # (batch_size, sequence_length, vocab_size)
            logit = logits[:, -1, :] # the last token's logits
            if temperature == 0:
                id_next = logit.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logit / temperature, dim=-1)
                id_next = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, id_next), dim=1)
            if save_logits:
                self.last_logits = logits
            yield id_next
    

class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim,hidden_dim=args.hidden_dim,dropout=args.dropout,)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self, x, kv_cache: Optional[KVCache] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention.forward(self.attention_norm(x), kv_cache=kv_cache, attn_mask=attn_mask)
        x = x + self.feed_forward.forward(self.ffn_norm(x))
        return x
    
_use_fast_attention = True
    
class Attention(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_kv_heads 
        self.dropout = args.dropout

        self.wq = nn.Linear(self.dim,self.n_heads*self.head_dim,bias=args.attn_bias)
            # one can batch up all heads in a big matrix
        self.wk = nn.Linear(self.dim,self.n_kv_heads*self.head_dim,bias=args.attn_bias)
            # one might have multiple heads sharing the same wk, wv,
        self.wv = nn.Linear(self.dim,self.n_kv_heads*self.head_dim,bias=args.attn_bias)
        self.wo = nn.Linear(self.n_heads*self.head_dim,self.dim,bias=False)

        self.resid_dropout = nn.Dropout(self.dropout)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.rope = RotaryEmbedding(self.head_dim, args.max_position_embeddings, args.rope_theta)
            
    def forward(self, x, kv_cache: Optional[KVCache] = None,attn_mask=None):
        bs, seq_len, dim = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim

        # compute the query, key, value vectors for each word vector
        q = self.wq(x).view(bs, seq_len, n_heads, head_dim)
            # (bs, seq_len, n_heads, head_dim)
        k = self.wk(x).view(bs, seq_len, n_kv_heads, head_dim)
        v = self.wv(x).view(bs, seq_len, n_kv_heads, head_dim)
            # (bs, seq_len, n_kv_heads, head_dim)

        if kv_cache is not None:
            k0, v0, seq_start = kv_cache.get()
        else:
            k0, v0, seq_start = None, None, 0

        # add positional information to q and k (explained later)
        t = torch.arange(seq_start, seq_start + seq_len, device=x.device)[None, :]
        q, k = self.rope(q, k, t)

        if kv_cache is not None:
            kv_cache.add(k.detach().clone(), v.detach().clone())
        if k0 is not None:
            k = torch.cat((k0, k), dim=1)
            v = torch.cat((v0, v), dim=1)

        # reshape the tensors for attention procedure
        q = q.transpose(1, 2) 
            # (bs, n_heads, seq_len, head_dim)
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=2).transpose(1, 2)
        v = v.repeat_interleave(n_heads // n_kv_heads, dim=2).transpose(1, 2)
            # repeats n_heads//n_kv_heads times along the third dimension (n_kv_heads)
            # (bs, n_heads, seq_len0 + seq_len, head_dim)
        if attn_mask is not None:
            attn_mask=attn_mask[:, None, :, :]
        
        if _use_fast_attention:
            output = F.scaled_dot_product_attention(q, k, v, 
                    is_causal=(seq_len>1), 
                        # the generated attn_mask is buggy when the matrix is not a square
                        # toggle is_casual at the presence of attn_mask to save half of the computation
                        # but flashattention dont support custom attn_mask at the moment. efficient attention is 10% slower
                    attn_mask=attn_mask,
                    dropout_p=(0.0 if not self.training else self.dropout)
                        # F do not bypass dropout when model.training is False
                    )
            # (bs, n_heads, seq_len, head_dim)
        else:
            # handwritten slow attention
            if attn_mask is None:
                attn_mask = torch.ones(1, 1, seq_len, seq_start+seq_len, device=x.device, dtype=torch.bool)
                attn_mask = torch.tril(attn_mask,diagonal=seq_start)
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim) # (bs, n_heads, seq_len, seq_len)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            scores = F.softmax(scores, dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v # (bs, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, n_heads*head_dim)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
            

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            # (head_dim//2,)
        self.register_buffer('freqs', freqs)

    def forward(self, q, k, t):
        input_dtype = q.dtype
        bs, seq_len, n_heads, head_dim = q.shape
        freqs = self.freqs[None, None, None, :].float()
            # need to use float32 in cos and sin
        freqs = torch.cat((freqs, freqs), dim=-1) 
        t = t[:, :, None, None].float()
        angles = freqs * t
        cos = angles.cos()
        sin = angles.sin()
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q.to(input_dtype), k.to(input_dtype)

def rotate_half(x):
    # Rotates half the hidden dims of the input.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x, input_dtype = x.float(), x.dtype
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)    
    
class KVCache:
    def __init__(self):
        self.k=None
        self.v=None
    def add(self, k, v):
        # (bs, seq_len=1, n_heads, head_dim)
        if self.k is None:
            self.k = k
            self.v = v 
        else:
            self.k = torch.cat((self.k, k), dim=1)
            self.v = torch.cat((self.v, v), dim=1)
    def get(self):
        seq_len = self.k.shape[1] if self.k is not None else 0
        return self.k, self.v, seq_len

class MoEGate(nn.Module):
    def __init__(self, dim, n_experts, k_experts):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k_experts = k_experts
        self.weight=nn.Parameter(torch.empty((n_experts, dim)))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        n_experts, k_experts = self.n_experts, self.k_experts
        logits = F.linear(x, self.weight)
        scores = F.softmax(logits, dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k_experts, dim=-1, sorted=False)
            # (..., k_experts), (..., k_experts)
        if k_experts>1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True)+1e-20)
        return topk_idx, topk_weight
            
class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts, k_experts, dropout):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, dropout)
            for _ in range(n_experts)])
        self.gate = MoEGate(dim, n_experts, k_experts)
        self.n_experts = n_experts
        self.k_experts = k_experts
    def forward(self, x):
        expert_idx, expert_weight = self.gate(x)
        print(expert_idx)
        orig_shape = x.shape
        n_tokens, dim = x.numel()//x.shape[-1], x.shape[-1]
        n_experts, k_experts = self.n_experts, self.k_experts
        x = x.view(n_tokens, dim)
        expert_idx=expert_idx.view(n_tokens, k_experts)
        expert_weight=expert_weight.view(n_tokens, k_experts)
        flat_expert_idx = expert_idx.view(n_tokens * k_experts)
        flat_expert_weight = expert_weight.view(n_tokens * k_experts)
        if self.training:
            x=x.repeat_interleave(k_experts,dim=0) # (n_tokens*k_experts, dim)
            y=torch.empty_like(x)
            for i,expert in enumerate(self.experts):
                mask=flat_expert_idx==i
                y[mask]=expert(x[mask])
            y = (y.view(n_tokens, k_experts, dim) * expert_weight.unsqueeze(-1)).sum(dim=-2)
            return y.view(orig_shape)
        else:
            # e.g. flat_expert_idx = [0,1(token0), 0,3(token1), 1,3(token2)]
            #      flat_expert_weight = [0.5, 0.5, 0.3, 0.7, 0.6, 0.4]
            #      flat_idx =       [0,2(expert0), 1,4(expert1), (expert2) 3,5(expert3)]
            #      flat_token_idx = [0,1(expert0), 0,2(expert1), (expert2) 1,2(expert3)]
            #      token_separators = [2, 4, 4, 6]
            #      expert 0 processes token flat_token_idx[start:end]=[0,1]
            #                         with weight flat_expert_weight[idx[start:end]]=[0.5, 0.3]
            y=torch.zeros_like(x) 
                # (n_tokens*k_experts, dim)
            flat_idx=flat_expert_idx.argsort()
                # the order of computation tasks
            flat_token_idx=flat_idx//k_experts
                # the token idx of each task
            token_separators=flat_expert_idx.bincount().cpu().numpy().cumsum(0)
                # the range of each expert's tasks
            for i, end in enumerate(token_separators):
                start = token_separators[i-1] if i>0 else 0
                if start<end:
                    current_expert=self.experts[i]
                    current_token_idx=flat_token_idx[start:end]
                    current_tokens=x[current_token_idx] # (end-start, dim)
                    current_weights=flat_expert_weight[flat_idx[start:end]] # (end-start)
                    current_expert_out=current_expert(current_tokens)*current_weights.unsqueeze(-1)
                    y.scatter_add_(
                        dim=0,
                        index=current_token_idx.view(-1, 1).repeat(1, dim),
                        src=current_expert_out)
            return y.view(orig_shape)
            


