
import torch
import torch.nn as nn

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        return self.cos_cached[:, :, :seq_len, ...].to(x.device), self.sin_cached[:, :, :seq_len, ...].to(x.device)
        

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(self.inv_freq.device)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

def apply_rotary_pos_emb(x, cos, sin):
    # Reshape x to [batch, seq_len, num_heads, head_dim]
    batch_size, seq_len, d_model = x.shape
    num_heads = d_model // (cos.shape[-1])
    x = x.view(batch_size, seq_len, num_heads, -1)
    
    # Adjust cos and sin for broadcasting
    cos = cos.view(1, seq_len, 1, -1)
    sin = sin.view(1, seq_len, 1, -1)
    
    # Apply rotary embeddings
    x_rotated = (x * cos) + (rotate_half(x) * sin)
    
    # Reshape back to original shape
    x_rotated = x_rotated.view(batch_size, seq_len, d_model)
    
    return x_rotated
