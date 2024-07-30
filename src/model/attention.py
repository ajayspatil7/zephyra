import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        return (self.cos_cached[:, :, :seq_len, ...].to(x.device),
                self.sin_cached[:, :, :seq_len, ...].to(x.device))

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.HIDDEN_SIZE
        self.num_heads = config.NUM_ATTENTION_HEADS
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.ATTENTION_PROBS_DROPOUT_PROB)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, tgt_len, 1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, tgt_len, 1, self.head_dim).transpose(1, 2)

        key_states = key_states.expand(-1, self.num_heads, -1, -1)
        value_states = value_states.expand(-1, self.num_heads, -1, -1)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        src_len = key_states.size(2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, (key_states, value_states)