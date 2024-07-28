import tiktoken
import re
from .specialTokens import ZephyraTokens


class ZephyraTokeniser:
    def __init__(self, vocab=None):
        # Define special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 4,
            "<|user|>": 5,
            "<|assistant|>": 6,
            "<|system|>": 7,
            "<|context|>": 8,
            "<|question|>": 9,
            "<|answer|>": 10,
            "<|rationale_start|>": 11,
            "<|rationale_end|>": 12,
            "[SEP]": 13,  # Added for CoQA
        }
        # Initialize vocabulary
        self.vocab = vocab if vocab else self.special_tokens.copy()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def add_tokens(self, new_tokens):
        for token in new_tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        # Improved tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token.lower() for token in tokens]  # Convert to lowercase

    def detokenize(self, token_ids):
        tokens = [self.inv_vocab.get(token_id, "<unk>") for token_id in token_ids]
        return " ".join(tokens)

    def encode(self, text, max_length=None, add_special_tokens=True, truncation=False):
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        if add_special_tokens:
            token_ids = [self.vocab["<s>"]] + token_ids + [self.vocab["</s>"]]

        if max_length and truncation:
            token_ids = token_ids[:max_length]
        elif max_length:
            token_ids = token_ids[:max_length - 1] + [self.vocab["</s>"]]
            token_ids += [self.vocab["<pad>"]] * (max_length - len(token_ids))

        return token_ids

    def decode(self, token_ids):
        if token_ids[0] == self.vocab["<s>"]:
            token_ids = token_ids[1:]
        if token_ids[-1] == self.vocab["</s>"]:
            token_ids = token_ids[:-1]
        return self.detokenize(token_ids)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self.inv_vocab.get(token_id, "<unk>") for token_id in token_ids]

    def pad_sequence(self, sequences, max_length=None):
        if not max_length:
            max_length = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            seq = seq[:max_length]
            seq += [self.vocab["<pad>"]] * (max_length - len(seq))
            padded_sequences.append(seq)
        return padded_sequences

    def getVocabSize(self):
        return self.vocab_size

    # New method to get token ID
    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

    # New method to get token from ID
    def convert_id_to_token(self, id):
        return self.inv_vocab.get(id, "<unk>")

