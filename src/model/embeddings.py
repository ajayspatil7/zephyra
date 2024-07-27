import torch
import torch.nn as nn
import math

class ZephyraEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # Initialize weights and apply final processing
        self.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        self.word_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        Create position IDs from input embeddings.
        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape (batch_size, sequence_length, hidden_size).
        Returns:
            torch.Tensor: Position IDs of shape (batch_size, sequence_length).
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.config.pad_token_id + 1, sequence_length + self.config.pad_token_id + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class ZephyraLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)