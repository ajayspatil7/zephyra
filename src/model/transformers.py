import torch
import torch.nn as nn
from attention import MultiHeadAttention
from src.model.encoding import PositionalEncoding


class TransformerBlock(nn.Module):

    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + self.dropout(attended))
        feedforward = self.ff(x)
        return self.norm2(x + self.dropout(feedforward))


class LanguageModel(nn.Module):


    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.model_name = "Zephyra Version 1"


    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc(x)
    

"""<-----------End of transformers.py----------->"""