import json
import random

def split_dataset(input_file, train_file, val_file, val_ratio=0.1, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the full dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError("Input data should be a list of examples")
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split index
    split_idx = int(len(data) * (1 - val_ratio))
    
    # Split the data
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save train dataset
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    
    # Save validation dataset
    with open(val_file, 'w') as f:
        json.dump(val_data, f)
    
    print(f"Total examples: {len(data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

# Usage
input_file = './data/dataset/coqa.json'
train_file = './data/dataset/coqa_train.json'
val_file = './data/dataset/coqa_val.json'

split_dataset(input_file, train_file, val_file, val_ratio=0.1)