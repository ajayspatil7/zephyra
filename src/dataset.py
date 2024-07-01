import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, seq_len, tokenizer):
        self.text = text
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return ( torch.tensor(self.data[idx:idx+self.seq_len]), torch.tensor(self.data[idx+1:idx+self.seq_len+1]) )

"""<-----------End of dataset.py----------->"""