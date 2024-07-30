import sys
import os
import warnings
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset, DataLoader

class CoQADataset(Dataset):
    def __init__(self, data_path, max_len):
        data = torch.load(data_path)
        
        # Check if data is a list or a dictionary
        if isinstance(data, dict):
            self.inputs = data['inputs']
            self.targets = data['targets']
        elif isinstance(data, list) and len(data) == 2:
            self.inputs, self.targets = data
        else:
            raise TypeError("Data should be a dictionary with 'inputs' and 'targets' keys or a list of two elements.")
        
        self.max_len = max_len
        
        print(type(data))  # To check the type of loaded data
        print(data.keys() if isinstance(data, dict) else "Data is a list")
        print(f"Inputs length {len(self.inputs)}")
        print(f"Targets length {len(self.targets)}")

        if not isinstance(self.inputs, list) or not isinstance(self.targets, list):
            raise TypeError("Inputs and targets should be lists.")
        
        if len(self.inputs) != len(self.targets):
            raise ValueError("Mismatch in length between inputs and targets.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_item = self.inputs[idx]
        target_item = self.targets[idx]
        
        # Convert to torch tensors
        input_item = torch.tensor(input_item, dtype=torch.long)
        target_item = torch.tensor(target_item, dtype=torch.long)

        return {
            'input': input_item,
            'target': target_item
        }

# Define the path to your dataset and max_len
data_path = './data/datasets/train.pt'
max_len = 512  # Adjust based on your model's expected input length

# Instantiate the dataset
dataset = CoQADataset(data_path, max_len)

# Create a DataLoader for iterating through the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Print out a few samples
print("\nDataset samples:")
for i, batch in enumerate(dataloader):
    print(f"\nSample {i+1}:")
    print("Input:", batch['input'])
    print("Target:", batch['target'])
    
    if i == 1:  # Print only the first 2 samples
        break

