import torch
import torch.nn as nn
from .embeddings import ZephyraEmbeddings
from .layers import ZephyraEncoder


class ZephyraPreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.INITIALIZER_RANGE)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.INITIALIZER_RANGE)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def tie_weights(self):
        if hasattr(self, "get_output_embeddings") and hasattr(self, "get_input_embeddings"):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        if getattr(self.config, "USE_TORCHSCRIPT", False):
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if hasattr(output_embeddings, "bias") and output_embeddings.bias is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def post_init(self):
        """
        Initialize weights and apply final processing if needed.
        """
        self.apply(self._init_weights)
        self.tie_weights()

class ZephyraModel(ZephyraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = ZephyraEmbeddings(config)
        self.encoder = ZephyraEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
    ):
#         print(f"ZephyraModel forward method - input_ids shape: {input_ids.shape}")
#         print(f"ZephyraModel forward method - attention_mask shape: {attention_mask.shape}")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.OUTPUT_HIDDEN_STATES
        )
        use_cache = use_cache if use_cache is not None else self.config.USE_CACHE
        return_dict = return_dict if return_dict is not None else self.config.USE_RETURN_DICT

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         print(f"ZephyraModel forward method - extended_attention_mask shape: {extended_attention_mask.shape}")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values[0][0].shape[2] if past_key_values is not None else 0,
        )

#         print(f"ZephyraModel forward method - embedding_output shape: {embedding_output.shape}")

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

#         print(f"ZephyraModel forward method - encoder_outputs type: {type(encoder_outputs)}")
#         if isinstance(encoder_outputs, dict):
#             for k, v in encoder_outputs.items():
#                 if isinstance(v, torch.Tensor):
#                     print(f"  {k} shape: {v.shape}")
#                 else:
#                     print(f"  {k} type: {type(v)}")

        return encoder_outputs
    
    
class ZephyraForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.zephyra = ZephyraModel(config)
        self.qa_outputs = nn.Linear(config.HIDDEN_SIZE, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.zephyra(input_ids, attention_mask=attention_mask)
        sequence_output = outputs['last_hidden_state']

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Flatten logits to match start/end positions
            start_logits = start_logits.view(-1, start_logits.size(-1))
            end_logits = end_logits.view(-1, end_logits.size(-1))
            
            # Flatten start and end positions
            start_positions = start_positions.view(-1)
            end_positions = end_positions.view(-1)
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {
            'loss': total_loss,
            'start_logits': start_logits,
            'end_logits': end_logits
        }