import torch
import sys
import os
import warnings
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings("ignore")

def inspect_pt_file(file_path):
    print(f"Inspecting file: {file_path}")
    
    try:
        data = torch.load(file_path)
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Number of items in the list: {len(data)}")
            
            if len(data) > 0:
                print("\nFirst item in the list:")
                first_item = data[0]
                if isinstance(first_item, dict):
                    for key, value in first_item.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: Tensor of shape {value.shape} and dtype {value.dtype}")
                        elif isinstance(value, list):
                            print(f"  {key}: List of length {len(value)}")
                        else:
                            print(f"  {key}: {type(value)}")
                else:
                    print(f"  Type: {type(first_item)}")
            
            print("\nKeys in all items:")
            all_keys = set()
            for item in data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            print(f"  {', '.join(all_keys)}")
        
        elif isinstance(data, dict):
            print("Keys in the dictionary:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor of shape {value.shape} and dtype {value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: List of length {len(value)}")
                else:
                    print(f"  {key}: {type(value)}")
        
        else:
            print(f"Unexpected data type: {type(data)}")
    
    except Exception as e:
        print(f"Error loading or inspecting file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pt_file.py <path_to_pt_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    inspect_pt_file(file_path)