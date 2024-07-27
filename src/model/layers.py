import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .attention import MultiQueryAttention

class ZephyraFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = self.get_activation(config.hidden_act)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_activation(self, activation_string):
        if activation_string == "gelu":
            return nn.GELU()
        elif activation_string == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_string}")

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class ZephyraLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = MultiQueryAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            attention_dropout=config.attention_probs_dropout_prob
        )
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.ffn = ZephyraFeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        # Self-attention
        attention_output, present_key_value = self.attention(
            hidden_states, 
            attention_mask=attention_mask, 
            past_key_value=past_key_value
        )
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(hidden_states + attention_output)

        # Feed-forward network
        if self.config.use_gradient_checkpointing and self.training:
            ffn_output = self.gradient_checkpointed_ffn(attention_output)
        else:
            ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_norm(attention_output + ffn_output)

        outputs = (ffn_output,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def gradient_checkpointed_ffn(self, attention_output):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        return checkpoint(create_custom_forward(self.ffn), attention_output)

class ZephyraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ZephyraLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
        }