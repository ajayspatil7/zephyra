import torch
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import config
from src.model.zephyra import ZephyraForQuestionAnswering

def inspect_model(model, file):
    file.write("Model Architecture:\n")
    file.write(str(model) + "\n")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    file.write("\nModel Parameters:\n")
    file.write(f"Total parameters: {total_params}\n")
    file.write(f"Trainable parameters: {trainable_params}\n")
    
    file.write("\nDetailed Parameter Information:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            file.write(f"{name}:\n")
            file.write(f"  Shape: {param.shape}\n")
            file.write(f"  Data type: {param.dtype}\n")
            file.write(f"  Device: {param.device}\n")
            file.write(f"  Requires grad: {param.requires_grad}\n")

def test_forward_pass(model, file):
    file.write("\nTesting Forward Pass:\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    start_positions = torch.randint(0, seq_length, (batch_size,)).to(device)
    end_positions = torch.randint(0, seq_length, (batch_size,)).to(device)
    
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        file.write("Forward pass successful!\n")
        file.write("Output keys: " + str(outputs.keys()) + "\n")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                file.write(f"{key} shape: {value.shape}\n")
            else:
                file.write(f"{key}: {value}\n")
    except Exception as e:
        file.write(f"Error during forward pass: {str(e)}\n")

if __name__ == "__main__":
    model = ZephyraForQuestionAnswering(config)
    with open("model_inspection.txt", "w") as file:
        inspect_model(model, file)
        test_forward_pass(model, file)

    print("Model inspection and forward pass test completed. Results are written to 'model_inspection.txt'.")
