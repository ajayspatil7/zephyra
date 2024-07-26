import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding module for positional encoding for self-attention mechanism.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Initialize the RotaryEmbedding module.

        Args:
            dim (int): The dimension of the input tensor.
            max_position_embeddings (int): The maximum sequence length for positional embeddings.
            base (int): The base value for the positional encoding calculation.
            device (torch.device): The device to store the tensors on.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        """
        Forward pass of the RotaryEmbedding module.

        Args:
            x (torch.Tensor): The input tensor.
            seq_len (int): The length of the sequence.

        Returns:
            tuple: A tuple containing the cosine and sine embeddings.
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

class RotaryAttention(nn.Module):
    """
    Rotary Attention module for self-attention mechanism.
    """

    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings=2048):
        """
        Initialize the RotaryAttention module.

        Args:
            hidden_size (int): The hidden size of the input tensor.
            num_attention_heads (int): The number of attention heads.
            max_position_embeddings (int): The maximum sequence length for positional embeddings.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass of the RotaryAttention module.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The output of the attention mechanism.
        """
        batch_size, seq_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=seq_length)
        q, k = applyRotaryPosEmbedding(q, k, cos, sin)

        if attention_mask is not None:
            # Reshape attention_mask to [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        attn_output = sdpAttention(q, k, v, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

def applyRotaryPosEmbedding(q, k, cos, sin):
    """
    Apply rotary positional embedding to the query and key tensors.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cosine embeddings.
        sin (torch.Tensor): The sine embeddings.

    Returns:
        tuple: A tuple containing the modified query and key tensors.
    """
    return (q * cos) + (rotateHalf(q) * sin), (k * cos) + (rotateHalf(k) * sin)

def rotateHalf(x):
    """
    Rotate the input tensor by half.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The rotated tensor.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def sdpAttention(q, k, v, mask=None):
    """
    Compute the scaled dot-product attention.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        v (torch.Tensor): The value tensor.
        mask (torch.Tensor): The attention mask tensor.

    Returns:
        torch.Tensor: The output of the attention mechanism.
    """
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits + mask
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values

