import torch
import sys
import os
import warnings
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')


def inspect_data_file(file_path):
    try:
        # Load the data from the .pt file
        data = torch.load(file_path)
        data_type = type(data)

        # Print some information about the data
        print(f"\nLoaded data from {file_path}")
        print(f"Type of data: {data_type}")
        
        if isinstance(data, list):
            data_len = len(data)
            print(f"Number of entries: {data_len}")
#             print(f"First entry example: {data[0]}")
        elif isinstance(data, dict):
            dict_keys = list(data.keys())
            
            key_zero = data[list(data.keys())[0]]
            key_one = data[list(data.keys())[1]]
            
            sample = key_zero[0]
            sample_type = type(sample)
            
            sample2 = key_zero[1]
            sample2_type = type(sample2)
            
            key_zero_len = len(data[list(data.keys())[0]])
            key_zero_type = type(data[list(data.keys())[0]])

            key_one_len =  len(data[list(data.keys())[1]])
            key_one_type = type(data[list(data.keys())[1]])
            
            print(f"Keys in the data: {dict_keys}")
            
            if isinstance(key_zero, list) and isinstance(key_one, list) :
                print(f"\nType of {dict_keys[0]} is a {key_zero_type} with len {key_zero_len}")
#                 print(f"Type of {dict_keys[0]} sample 1 {sample_type} with value {sample} and sample 1 len {len(sample)}")
                print(f"Type of {dict_keys[0]} sample 1 {sample_type} with sample 1 len {len(sample)}")
                
                print(f"\nType of {dict_keys[1]} is a {key_one_type} with len {key_one_len}")
#                 print(f"Type of {dict_keys[1]} sample 1 {sample2_type} with value {sample2} and sample 1 len {len(sample2)}")
                print(f"Type of {dict_keys[1]} sample 1 {sample2_type} with sample 1 len {len(sample2)}")
                
            if isinstance(data[list(data.keys())[1]], dict):
                print(f"Type of {type(data[list(data.keys())[1]].keys())} with len {list(data[list(data.keys())[0]].keys())}")
#             print(f"Example entry: {data}")
        else:
            print("Unexpected data format.")
        
        # Print a few samples for verification
#         if isinstance(data, list) and len(data) > 0:
#             print("\nSample entries:")
#             for i in range(min(1, len(data))):  # Print up to 5 samples
#                 print(f"Sample {i}: {data[i]}")
#         elif isinstance(data, dict):
#             print("\nSample entry:")
#             for key, value in data.items():
#                 print(f"Key: {key}, Value: {value}")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Define paths to your .pt files
    train_data_path = './data/datasets/train.pt'
    val_data_path = './data/datasets/dev.pt'
    
    print("Inspecting training data...")
    inspect_data_file(train_data_path)
    
    print("\nInspecting validation data...")
    inspect_data_file(val_data_path)

if __name__ == "__main__":
    main()
