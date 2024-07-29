import re


class ZephyraTokenizer:
    def __init__(self, vocab, special_tokens):
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.special_tokens = special_tokens
        self.special_token_ids = {self.vocab[token]: token for token in special_tokens.values()}
    
    def encode(self, text):
        tokens = []
        for word in re.findall(r'\b\w+\b|\S', text.lower()):
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab[self.special_tokens["UNK"]])
        return tokens
    
    def decode(self, token_ids):
        return " ".join([self.inv_vocab[token_id] for token_id in token_ids if token_id in self.inv_vocab])
    
    def add_special_tokens(self, text, token_type):
        if token_type in self.special_tokens:
            return f"{self.special_tokens[token_type]} {text} {self.special_tokens['EOS']}"
        return text
