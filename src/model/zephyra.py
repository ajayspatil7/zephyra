import torch
import torch.nn as nn
from .embeddings import ZephyraEmbeddings
from .layers import ZephyraLayer

class ZephyraModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, max_position_embeddings, dropout_rate=0.1):
        super().__init__()
        self.embeddings = ZephyraEmbeddings(vocab_size, hidden_size, max_position_embeddings, dropout_rate)
        self.layers = nn.ModuleList([
            ZephyraLayer(hidden_size, num_attention_heads, intermediate_size, attention_dropout=dropout_rate, hidden_dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, past_key_values=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)
        
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = ()
        all_attentions = ()
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            all_hidden_states += (hidden_states,)
            
            layer_outputs, past_key_value = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                past_key_value=past_key_value
            )
            hidden_states = layer_outputs

            all_attentions += (layer_outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }

class ZephyraForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, max_position_embeddings, num_labels, dropout_rate=0.1):
        super().__init__()
        self.zephyra = ZephyraModel(vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, max_position_embeddings, dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None):
        outputs = self.zephyra(input_ids, attention_mask, token_type_ids, position_ids)
        pooled_output = outputs["last_hidden_state"][:, 0, :]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss,"logits": logits,"hidden_states": outputs.get("hidden_states"),"attentions": outputs.get("attentions"),}