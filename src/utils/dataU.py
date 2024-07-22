""" This file contains basisc utility functions  """
import torch
from torch.utils.data import Dataset
import json
import config
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
        
    

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)


    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(item['target_ids'], dtype=torch.long)

        # Pad input_ids if necessary
        if len(input_ids) < self.max_length:
            padding = torch.full((self.max_length - len(input_ids),), self.tokenizer.get_pad_token_id(), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        else:
            input_ids = input_ids[:self.max_length]

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.get_pad_token_id()).float()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }

    @staticmethod
    def collate_fn(batch):
        # Determine max length in the batch for labels
        max_label_length = max(len(item['labels']) for item in batch)

        # Pad labels to max length in the batch
        for item in batch:
            labels = item['labels']
            padding = torch.full((max_label_length - len(labels),), -100, dtype=torch.long)
            item['labels'] = torch.cat([labels, padding])

        # Stack all tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Usage example:
if __name__ == "__main__":
    tokenizer = ZephyraTokenizer()
    dataset = ZephyraCoQADataset('preprocessed_coqa.json', tokenizer)
    print(f"Dataset size: {len(dataset)}")

    # Test a single item
    item = dataset[0]
    print(f"Input IDs shape: {item['input_ids'].shape}")
    print(f"Attention Mask shape: {item['attention_mask'].shape}")
    print(f"Labels shape: {item['labels'].shape}")

    # Test batch collation
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=ZephyraCoQADataset.collate_fn)
    batch = next(iter(dataloader))
    print(f"Batch Input IDs shape: {batch['input_ids'].shape}")
    print(f"Batch Attention Mask shape: {batch['attention_mask'].shape}")
    print(f"Batch Labels shape: {batch['labels'].shape}")