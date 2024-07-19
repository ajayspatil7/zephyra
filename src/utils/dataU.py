""" This file contains basisc utility functions  """
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        encoding = self.tokenizer.encode(line, add_special_tokens=True)
        
        if len(encoding) > self.max_length:
            encoding = encoding[:self.max_length]
        else:
            pad_token_id = self.tokenizer.get_pad_token_id()
            encoding += [pad_token_id] * (self.max_length - len(encoding))
        
        return torch.tensor(encoding, dtype=torch.long)
