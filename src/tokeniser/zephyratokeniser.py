import re
import sys
import os
import warnings
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')


class ZephyraTokeniser:
    def __init__(self, vocab, special_tokens_class):
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.special_tokens = special_tokens_class
        
        # Ensure special tokens are in vocab with correct names
        for attr, token in vars(special_tokens_class).items():
            if isinstance(token, str) and token.startswith('<') and token.endswith('>'):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text):
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab[self.special_tokens.UNK])
        return tokens

    def decode(self, token_ids):
        return ' '.join([self.inv_vocab.get(token_id, self.special_tokens.UNK) for token_id in token_ids])

    def add_special_tokens(self, text, token_type):
        return f"{token_type} {text} {self.special_tokens.EOS}"

    def encode_plus(self, question, context, max_length=512, padding='max_length', truncation=True):
        question_tokens = self.encode(self.add_special_tokens(question, self.special_tokens.QUESTION))
        context_tokens = self.encode(self.add_special_tokens(context, self.special_tokens.CONTEXT))
        
        combined_tokens = question_tokens + context_tokens
        if truncation and len(combined_tokens) > max_length:
            combined_tokens = combined_tokens[:max_length]
        
        attention_mask = [1] * len(combined_tokens)
        
        if padding == 'max_length':
            pad_length = max_length - len(combined_tokens)
            combined_tokens += [self.vocab[self.special_tokens.PAD]] * pad_length
            attention_mask += [0] * pad_length
        
        return {
            "input_ids": torch.tensor([combined_tokens]),
            "attention_mask": torch.tensor([attention_mask])
        }

    def print_vocab_info(self):
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"First 10 tokens: {list(self.vocab.keys())[:10]}")
        print(f"Last 10 tokens: {list(self.vocab.keys())[-10:]}")
        print(f"Special tokens: {vars(self.special_tokens)}")
