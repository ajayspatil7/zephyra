import torch

# Define file paths
train_inputs_path = './data/dataset/train_inputs.pt'
train_targets_path = './data/dataset/train_targets.pt'
dev_inputs_path = './data/dataset/dev_inputs.pt'
dev_targets_path = './data/dataset/dev_targets.pt'

train_output_path = './data/dataset/train.pt'
dev_output_path = './data/dataset/dev.pt'

def combine_data(inputs_path, targets_path, output_path):
    # Load inputs and targets
    inputs = torch.load(inputs_path)
    targets = torch.load(targets_path)
    
    # Check if the lengths match
    if len(inputs) != len(targets):
        raise ValueError(f"Length mismatch: inputs ({len(inputs)}) and targets ({len(targets)})")
    
    # Combine into a dictionary
    combined = {'inputs': inputs, 'targets': targets}
    
    # Save combined data to the output file
    torch.save(combined, output_path)
    print(f"Combined data saved to {output_path}")

# Combine train data
combine_data(train_inputs_path, train_targets_path, train_output_path)

# Combine dev data
combine_data(dev_inputs_path, dev_targets_path, dev_output_path)
