""" This file contains basisc utility functions  """
import torch
from torch.utils.data import Dataset
import json
from src import config
from src.tokenizer import ZephyraTokenizer

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



class ZephyraCoQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(item['target_ids'], dtype=torch.long)

        # Truncate input_ids if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }
    
    def get_pad_token_id(self):
        return self.tokenizer.getPaddingTokenId()

    @staticmethod
    def collate_fn(batch):
        # Find max lengths
        max_input_len = max(len(item['input_ids']) for item in batch)
        max_label_len = max(len(item['labels']) for item in batch)

        # Pad sequences
        input_ids = [item['input_ids'].tolist() + [0] * (max_input_len - len(item['input_ids'])) for item in batch]
        labels = [item['labels'].tolist() + [-100] * (max_label_len - len(item['labels'])) for item in batch]

        # Create attention masks
        attention_mask = [[1] * len(item['input_ids']) + [0] * (max_input_len - len(item['input_ids'])) for item in batch]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Usage example:
if __name__ == "__main__":
    tokenizer = ZephyraTokenizer()
    dataset = ZephyraCoQADataset('./data/dataset/coqa_train.json', tokenizer)
    print(f"Dataset size: {len(dataset)}")

    # Test a single item
    item = dataset[0]
    print(f"Input IDs shape: {item['input_ids'].shape}")
    print(f"Labels shape: {item['labels'].shape}")

    # Test batch collation
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=ZephyraCoQADataset.collate_fn)
    batch = next(iter(dataloader))
    print(f"Batch Input IDs shape: {batch['input_ids'].shape}")
    print(f"Batch Attention Mask shape: {batch['attention_mask'].shape}")
    print(f"Batch Labels shape: {batch['labels'].shape}")