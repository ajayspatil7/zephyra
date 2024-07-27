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
        """
        Compute and cache the cos and sin values for rotary embeddings.
        """
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        """
        Apply rotary embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            seq_len (int, optional): Sequence length. If None, uses x.shape[1].

        Returns:
            tuple: (cos, sin) tensors for rotary embedding computation.
        """
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

def rotate_half(x):
    """
    Rotate half of the dimensions of the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with half of its dimensions rotated.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embeddings to queries and keys.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine values for rotary embedding.
        sin (torch.Tensor): Sine values for rotary embedding.

    Returns:
        tuple: (q_rotated, k_rotated) tensors with rotary embeddings applied.
    """
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


class MultiQueryAttention(nn.Module):
    
    def __init__(self, hidden_size, num_heads, max_position_embeddings=2048, attention_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize model parameters.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        """
        Compute multi-query attention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, 1, seq_len).
            past_key_value (tuple, optional): Cached key and value tensors for incremental decoding.

        Returns:
            tuple: (attn_output, (key, value)) where attn_output is the output of the attention layer
                   and (key, value) are the updated key and value for caching.
        """
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, 1, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        cos, sin = self.rotary_emb(v, seq_len=k.shape[2])
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, (k, v)