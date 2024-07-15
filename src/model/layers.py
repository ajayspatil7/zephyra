import torch.nn as nn
from .attention import RotaryAttention

class ZephyraBlock(nn.Module):

    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, layer_norm_eps=1e-5):
        super().__init__()
        self.attention = RotaryAttention(hidden_size, num_attention_heads)
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(self.ln1(x), attention_mask)
        x = x + attn_output
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        return x
