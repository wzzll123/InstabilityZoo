import transformer_engine.pytorch as te
import torch
from torch.nn import functional as F
import math

def cal_multihead_attn_entropy(q,k):
    assert q.dim() == 4, 'shape of q should be (B, nh, T, hs)'
    assert k.dim() == 4, 'shape of k should be (B, nh, T, hs)'
    T = q.shape[-2]
    mask = torch.tril(torch.ones(T, T).view(1, 1, T, T)).to(q.device)
    with torch.no_grad():
        with torch.autocast('cuda', torch.bfloat16):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            max_attention_logit = att.max() # Store the max logit
            mean_attention_logit = att.mean() # Store the max logit
            min_attention_logit = att.min() # Store the max logit
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            entropy = torch.mean(-torch.nansum(att*torch.log(att), dim=-1))
    return max_attention_logit, mean_attention_logit, min_attention_logit, entropy

def entropy_hook_for_layernormqkv(module, input, output):
    # after te.layer_norm_qkv
    assert output.dim() == 3, 'shape of output should be (B, T, 3*embd)'
    B,T = output.shape[0], output.shape[1]
    n_embd = output.shape[2] // 3
    q, k, v  = output.split(n_embd, dim=2)
    if n_embd == 768:
        n_head = 12
    elif n_embd == 1280:
        n_head =20
    elif n_embd == 1600:
        n_head = 25
    k = k.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, n_head, n_embd // n_head).transpose(1, 2)
    module.max_attention_logit, module.entropy =  cal_multihead_attn_entropy(q,k)

def entropy_hook(module, input, output):
    assert module.qkv_format == 'bshd'
    q, k = input[0], input[1]
    module.max_attention_logit, module.mean_attention_logit, module.min_attention_logit, module.entropy = cal_multihead_attn_entropy(q.transpose(1,2),k.transpose(1,2))

