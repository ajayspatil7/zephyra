import torch
import torch.nn as nn
import math
import re
from collections import defaultdict

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    

    # TODO: BPE (byte pair encoding) should be implemented here


class BytePairEncoding:


    def __init__(self, data: str, itter: int):
        self.data = data
        self.itter = itter
        self.vocab = self.getVocab()

    def encode(self) -> dict:
        """
        Performs byte pair encoding on the input data.
        :returns dict with the word and their respective token
        """
        for _ in range(self.itter):
            pairs = self.getPair()
            best = max(pairs, key=pairs.get)
            self.vocab = self.mergeVocab(best)
        return self.vocab

    def getVocab(self) -> dict:
        """
        :returns words in the large string and their occurrence count
        """
        vocab = defaultdict(int)
        for l in self.data.split('\n'):
            for w in l.split():
                vocab[' '.join(list(w)) + '</w>'] += 1
        return vocab

    def getPair(self) -> dict:
        """
        :returns the possible pair based on frequent occurrence
        """
        pairs = defaultdict(int)
        for w, o in self.vocab.items():
            symbols = w.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += o
        return pairs

    def mergeVocab(self, pair: tuple) -> dict:
        """
        :pair -> pair of characters
        :returns new vocab with word and where the repeated occurrence occurred
        """
        vocab_out = {}
        tokens = re.escape(' '.join(pair))
        expression = re.compile(r'(?<!\S)' + tokens + r'(?!\S)')
        for word in self.vocab:
            return_word = expression.sub(''.join(pair), word)
            vocab_out[return_word] = self.vocab[word]
        return vocab_out



"""<-----------End of encodings.py----------->"""