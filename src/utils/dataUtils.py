""" This file contains basisc utility functions  """
import torch
from torch.utils.data import Dataset
import json
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
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        labels = torch.tensor(item['target_ids'], dtype=torch.long)

        # Pad or truncate input_ids and labels
        input_ids = self._pad_or_truncate(input_ids)
        labels = self._pad_or_truncate(labels)

        attention_mask = (input_ids != self.tokenizer.getPaddingTokenId()).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _pad_or_truncate(self, tensor):
        if len(tensor) < self.max_length:
            return torch.cat([tensor, torch.full((self.max_length - len(tensor),), self.tokenizer.getPaddingTokenId(), dtype=torch.long)])
        else:
            return tensor[:self.max_length]

    def get_pad_token_id(self):
        return self.tokenizer.getPaddingTokenId()

    @staticmethod
    def collate_fn(batch):
        # Find max length in the batch
        max_len = max(len(item['input_ids']) for item in batch)

        # Pad all tensors to max_len
        input_ids = torch.stack([torch.cat([item['input_ids'], torch.full((max_len - len(item['input_ids']),), item['input_ids'][-1], dtype=torch.long)]) for item in batch])
        attention_mask = torch.stack([torch.cat([item['attention_mask'], torch.zeros(max_len - len(item['attention_mask']), dtype=torch.long)]) for item in batch])
        labels = torch.stack([torch.cat([item['labels'], torch.full((max_len - len(item['labels']),), -100, dtype=torch.long)]) for item in batch])

        # print(f"Collated shapes - Input: {input_ids.shape}, Attention: {attention_mask.shape}, Labels: {labels.shape}")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }



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