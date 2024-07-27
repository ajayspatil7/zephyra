import torch.nn as nn
from .attention import MultiQueryAttention
import torch


class ZephyraFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_function=nn.GELU(), dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = activation_function
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class ZephyraLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_dropout=0.1, hidden_dropout=0.1):
        super().__init__()
        self.attention = MultiQueryAttention(hidden_size, num_attention_heads, attention_dropout=attention_dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(hidden_dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        self.ffn = ZephyraFeedForward(hidden_size, intermediate_size, dropout_rate=hidden_dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        attention_output, past_key_value = self.attention(hidden_states, attention_mask, past_key_value)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(hidden_states + attention_output)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_norm(attention_output + ffn_output)

        return ffn_output, past_key_value
