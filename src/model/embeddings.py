import torch
import torch.nn as nn
import math

class ZephyraEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_SIZE, padding_idx=config.PAD_TOKEN_ID)
        self.position_embeddings = nn.Embedding(config.MAX_POSITION_EMBEDDINGS, config.HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(config.TYPE_VOCAB_SIZE, config.HIDDEN_SIZE)

        self.LayerNorm = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.dropout = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

        self.register_buffer("position_ids", torch.arange(config.MAX_POSITION_EMBEDDINGS).expand((1, -1)))

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.word_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        nn.init.normal_(self.token_type_embeddings.weight, std=0.02)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
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

class ZephyraLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)