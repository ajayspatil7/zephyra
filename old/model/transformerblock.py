import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.encoding import PositionalEncoding

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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + self.dropout1(attended))
        feedforward = self.ff(x)
        return self.norm2(x + self.dropout2(feedforward))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(self.embedding.weight.device) # Update: Fixed error causing uneven distributions of tensors on cuda and CPU
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Zephyra(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)

    def generate(self, start_tokens, max_length, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device  # Update: Fixed error causing uneven distributions of tensors on cuda and CPU
        current_tokens = start_tokens.to(device)


        with torch.no_grad():
            for _ in range(max_length - len(start_tokens)):
                output = self(current_tokens)
                logits = output[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                if next_token.item() == self.encoder.embedding.num_embeddings - 1:  # End token
                    break

        return current_tokens

    def generate_answer(self, question, tokenizer, max_length=512, temperature=0.5):
        device = next(self.parameters()).device  
        question_tokens = torch.tensor(tokenizer.encode("Question: " + question + " Answer:")).unsqueeze(0)
        generated_tokens = self.generate(question_tokens, max_length, temperature)
        return tokenizer.decode(generated_tokens[0].tolist()).split("Answer:", 1)[1].strip()
    

    
    """<-----------End of transformerBlock.py----------->"""