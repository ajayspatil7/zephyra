import torch
import argparse
import os

def inspect_checkpoint(checkpoint_path):
    print(f"Inspecting checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Check if it's a state_dict or a dictionary containing a state_dict
    if isinstance(checkpoint, dict):
        print("Checkpoint is a dictionary containing multiple elements.")
        for key, value in checkpoint.items():
            if key == 'model_state_dict' or key == 'state_dict':
                print(f"\nModel State Dictionary ('{key}'):")
                inspect_state_dict(value)
            elif key == 'optimizer_state_dict':
                print("\nOptimizer State Dictionary:")
                inspect_optimizer_state_dict(value)
            elif isinstance(value, dict):
                print(f"\nNested Dictionary ('{key}'):")
                inspect_nested_dict(value)
            else:
                print(f"\n{key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"Shape: {value.shape}")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"Value: {value}")
    else:
        print("Checkpoint contains only the model state dictionary.")
        inspect_state_dict(checkpoint)

def inspect_state_dict(state_dict):
    total_params = 0
    for name, param in state_dict.items():
        print(f"Layer: {name}, Shape: {param.shape}, dtype: {param.dtype}")
        total_params += param.numel()
    print(f"Total number of parameters: {total_params}")

def inspect_optimizer_state_dict(optimizer_dict):
    print(f"Optimizer's state keys: {optimizer_dict.keys()}")
    if 'state' in optimizer_dict:
        print("Optimizer state:")
        for param_id, param_state in optimizer_dict['state'].items():
            print(f"  Parameter ID: {param_id}")
            for state_name, state_value in param_state.items():
                print(f"    {state_name}: {type(state_value)}")
                if hasattr(state_value, 'shape'):
                    print(f"      Shape: {state_value.shape}")

def inspect_nested_dict(nested_dict, indent="  "):
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            inspect_nested_dict(value, indent + "  ")
        else:
            print(f"{indent}{key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"{indent}  Shape: {value.shape}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"{indent}  Value: {value}")

def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint (.pt) files")
    parser.add_argument("checkpoint_path", type=str, help="Path to the .pt checkpoint file")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: File not found: {args.checkpoint_path}")
        return

    inspect_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    main()