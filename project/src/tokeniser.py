# create_tokenizer.py

import json
from tokenization.bytepairencoding import BPETokenizer
from utills.config import VOCAB_SIZE, TRAIN_DATA_PATH, TOKENIZER_PATH

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def main():
    print("Loading corpus...")
    corpus = load_corpus(TRAIN_DATA_PATH)

    print(f"Training BPE tokenizer with vocab size {VOCAB_SIZE}...")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(corpus)

    print(f"Saving tokenizer to {TOKENIZER_PATH}...")
    tokenizer.save(TOKENIZER_PATH)

    print("Tokenizer created and saved successfully!")

if __name__ == "__main__":
    main()
