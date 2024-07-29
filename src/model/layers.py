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
        self.attention_output = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.attention_dropout = nn.Dropout(config.HIDDEN_DROPOUT_PROB)
        self.attention_norm = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        
        self.ffn = ZephyraFeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.use_gradient_checkpointing = config.USE_GRADIENT_CHECKPOINTING if hasattr(config, 'USE_GRADIENT_CHECKPOINTING') else False

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
#         print(f"ZephyraLayer input hidden_states shape: {hidden_states.shape}")
        attention_output, present_key_value = self.attention(hidden_states, attention_mask, past_key_value)
#         print(f"ZephyraLayer attention_output shape: {attention_output.shape}")
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(hidden_states + attention_output)

        if self.use_gradient_checkpointing and self.training:
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
        self.layers = nn.ModuleList([ZephyraLayer(config) for _ in range(config.NUM_HIDDEN_LAYERS)])

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