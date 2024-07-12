# dataset.py

import torch
from torch.utils.data import Dataset
from tokenization.bytepairencoding import BPETokenizer

class ZephyraDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        tokens = self.tokenizer.encode(line)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.vocab['</w>']] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens)

def create_dataloaders(train_path, val_path, tokenizer_path, batch_size, max_length):
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    train_dataset = ZephyraDataset(train_path, tokenizer, max_length)
    val_dataset = ZephyraDataset(val_path, tokenizer, max_length)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader
