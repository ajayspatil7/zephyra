# src/model/zephyra.py
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

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return logits, loss