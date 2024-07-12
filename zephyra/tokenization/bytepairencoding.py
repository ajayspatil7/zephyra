import regex as re
from collections import defaultdict
import json
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def train(self, texts: List[str]):
        # Count initial vocabulary (characters)
        word_freqs = defaultdict(int)
        for text in texts:
            words = re.findall(self.pat, text)
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1

        # Initialize vocabulary with characters
        self.vocab = {char: i for i, char in enumerate(set(''.join(word_freqs.keys())))}
        self.vocab['</w>'] = len(self.vocab)
        
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair, word_freqs)
            self.merges.append(best_pair)
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        replacement = ''.join(pair)
        
        for word in list(word_freqs.keys()):
            new_word = pattern.sub(replacement, word)
            if new_word != word:
                word_freqs[new_word] = word_freqs.pop(word)

    def encode(self, text: str) -> List[int]:
        words = re.findall(self.pat, text)
        tokens = []
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            while len(word) > 0:
                subword = word
                while subword not in self.vocab and len(subword) > 0:
                    subword = subword[:-1]
                if subword == '':
                    # Unknown character, skip it
                    word = word[1:]
                else:
                    tokens.append(self.vocab[subword])
                    word = word[len(subword):]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return ''.join(self.inverse_vocab[token].replace(' ', '').replace('</w>', ' ') for token in tokens).strip()

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': self.merges
            }, f)

    def load(self, path: str):
        with open("/Users/ajay/Downloads/zephyra/project/src/tokenizer.json", 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

