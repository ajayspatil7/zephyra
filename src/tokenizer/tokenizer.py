
import tiktoken
from .specialTokens import ZephyraTokens

class ZephyraTokenizer:
    def __init__(self, model_name="cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.special_tokens = ZephyraTokens()
        self._add_special_tokens()
        self.vocab_size = self.tokenizer.n_vocab

    def _add_special_tokens(self):
        for token in vars(self.special_tokens).values():
            if token not in self.tokenizer.encode(token):
                self.tokenizer.add_special_tokens([token])

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            text = f"{self.special_tokens.BOS} {text} {self.special_tokens.EOS}"
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_vocab_size(self):
        return self.vocab_size

    def get_pad_token_id(self):
        return self.tokenizer.encode(self.special_tokens.PAD)[0]
