import torch
from torch.utils.data import Dataset

class QADataset(Dataset):


    def __init__(self, file_path, tokenizer, seq_len):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer
        self.seq_len = seq_len


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        text = self.data[idx].strip()
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [0] * (self.seq_len - len(tokens))  # Pad with 0
        
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])
    
