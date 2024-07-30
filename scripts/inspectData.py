import torch
import os
import sys
import warnings

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings("ignore")

from src.config import config
from src.utils.trainingUtils import CoQADataset

def analyze_dataset(dataset_path, max_len, vocab_size):
    print(f"Analyzing dataset: {dataset_path}")
    dataset = CoQADataset(dataset_path, max_len, vocab_size)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Max sequence length: {max_len}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Analyze a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        for key, value in sample.items():
            print(f"  {key}:")
            print(f"    Shape: {value.shape}")
            print(f"    Data type: {value.dtype}")
            print(f"    Min value: {value.min().item()}")
            print(f"    Max value: {value.max().item()}")
            if key == 'input_ids':
                print(f"    Unique tokens: {len(torch.unique(value))}")
            if key == 'target_ids':
                print(f"    Values: {value.tolist()}")

    # Analyze overall statistics
    all_input_lengths = []
    all_target_lengths = []
    for i in range(len(dataset)):
        sample = dataset[i]
        all_input_lengths.append(len(sample['input_ids']))
        all_target_lengths.append(len(sample['target_ids']))
    
    print("\nOverall statistics:")
    print(f"  Input lengths:")
    print(f"    Min: {min(all_input_lengths)}")
    print(f"    Max: {max(all_input_lengths)}")
    print(f"    Average: {sum(all_input_lengths) / len(all_input_lengths):.2f}")
    print(f"  Target lengths:")
    print(f"    Min: {min(all_target_lengths)}")
    print(f"    Max: {max(all_target_lengths)}")
    print(f"    Average: {sum(all_target_lengths) / len(all_target_lengths):.2f}")

if __name__ == "__main__":
    print("Analyzing training dataset:")
    analyze_dataset(config.TRAIN_PATH, config.MAX_LEN, config.VOCAB_SIZE)
    print("\nAnalyzing validation dataset:")
    analyze_dataset(config.VAL_PATH, config.MAX_LEN, config.VOCAB_SIZE)