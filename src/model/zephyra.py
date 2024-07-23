import torch.nn as nn
from .layers import ZephyraBlock
import torch.nn.functional as F

class ZephyraModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ZephyraBlock(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Print shape for debugging
        # print(f"Model output shape: {logits.shape}")
        
        return logits