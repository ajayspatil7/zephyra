class Tokenizer:

    def __init__(self, text):
        # Create vocabulary from the text
        self.vocab = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])


def create_tokenizer(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return Tokenizer(text)
    
"""<-----------End of tokeniser.py----------->"""