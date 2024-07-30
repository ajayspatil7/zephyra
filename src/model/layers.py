import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .attention import MultiQueryAttention

class ZephyraFeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.HIDDEN_SIZE, config.INTERMEDIATE_SIZE)
        self.fc2 = nn.Linear(config.INTERMEDIATE_SIZE, config.HIDDEN_SIZE)
        self.activation = nn.GELU() if config.HIDDEN_ACT == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class ZephyraLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiQueryAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.INTERMEDIATE_SIZE),
            nn.GELU(),
            nn.Linear(config.INTERMEDIATE_SIZE, config.HIDDEN_SIZE)
        )
        self.attention_norm = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.ffn_norm = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.dropout = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        attention_output, present_key_value = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            past_key_value
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        
        ffn_output = self.feed_forward(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(ffn_output)

        return hidden_states, present_key_value

class ZephyraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([ZephyraLayer(config) for _ in range(config.NUM_HIDDEN_LAYERS)])

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_present_key_values = () if use_cache else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_key_value = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value
            )

            if use_cache:
                all_present_key_values = all_present_key_values + (present_key_value,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": all_present_key_values,
            "hidden_states": all_hidden_states,
        }