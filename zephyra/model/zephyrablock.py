import torch
import torch.nn as nn
from ..attention.mq_attention import MultiQueryAttention
from ..embeddings.rotaryembeddings import RotaryEmbedding, apply_rotary_pos_emb
from ..layers.feedforward import SwiGLUFFN
from ..layers.normalization import RMSNorm

class ZephyraBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiQueryAttention(d_model, num_heads, dropout)
        self.feed_forward = SwiGLUFFN(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(dim=d_model // num_heads)
        print(f"ZephyraBlock RotaryEmbedding inv_freq shape: {self.rotary_emb.inv_freq.shape}")


    def forward(self, x, mask=None):
        # Apply rotary positional embeddings
        seq_len = x.size(1)
        cos, sin = self.rotary_emb(x, seq_len=seq_len)
        x = apply_rotary_pos_emb(x, cos, sin)

        # Attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # Feed Forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x
