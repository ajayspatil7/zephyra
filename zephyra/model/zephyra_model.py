# zephyra_model.py

import torch
import torch.nn as nn
from typing import Optional


from .zephyrablock import ZephyraBlock
from ..tokeniser import BPETokenizer

class ZephyraResolve(nn.Module):


    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1,
        tokenizer_path: Optional[str] = "",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length


        # Tokenizer
        self.tokenizer = BPETokenizer()
        if tokenizer_path:
            self.tokenizer.load(tokenizer_path)

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ZephyraBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()

        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embedding
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.layer_norm(x)

        # Output layer
        logits = self.output_layer(x)

        return logits


    def generate(self, input_ids: torch.Tensor, max_length: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        for _ in range(max_length - seq_length):
            # Get the last token's logits
            logits = self(input_ids)[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Break if we generate the EOS token
            if next_token.item() == self.tokenizer.vocab['</w>']:
                break

        return input_ids


    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
    

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens.tolist())