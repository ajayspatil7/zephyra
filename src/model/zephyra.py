import torch.nn as nn
from .layers import ZephyraBlock
import torch.nn.functional as F

class ZephyraModel(nn.Module):
    """
    ZephyraModel inherits from PyTorch nn module that represents the Zephyra model.

    Args:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The size of the hidden layer.
        num_layers (int): The number of layers in the model.
        num_attention_heads (int): The number of attention heads in the model.
        intermediate_size (int): The size of the intermediate layer.

    Attributes:
        embed (nn.Embedding): The embedding layer.
        layers (nn.ModuleList): The list of ZephyraBlocks.
        ln_f (nn.LayerNorm): The layer normalization layer.
        lm_head (nn.Linear): The linear layer for language modeling.

    """

    def __init__(self, vocab_size=100277, hidden_size=512, num_layers=8, num_attention_heads=8, intermediate_size=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ZephyraBlock(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the ZephyraModel.

        Args:
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, sequence_length, vocab_size).

        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits