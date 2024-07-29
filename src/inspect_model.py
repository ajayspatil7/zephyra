import sys
import os
import torch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import ZephyraConfig, config as config_dict
from src.model.zephyra import ZephyraForQuestionAnswering
from src.tokenizers import ZephyraTokenizer

def inspect_model():
    print("Initializing Zephyra model for inspection...")
    
    # Create the config object
    config = ZephyraConfig(**config_dict.__dict__)
    
    # Initialize tokenizer
    tokenizer = ZephyraTokenizer()
    
    # Update vocabulary size in config
    config.VOCAB_SIZE = tokenizer.getVocabSize()
    
    # Print some config values
    print(f"\nConfig values:")
    print(f"VOCAB_SIZE: {config.VOCAB_SIZE}")
    print(f"HIDDEN_SIZE: {config.HIDDEN_SIZE}")
    print(f"NUM_HIDDEN_LAYERS: {config.NUM_HIDDEN_LAYERS}")
    print(f"NUM_ATTENTION_HEADS: {config.NUM_ATTENTION_HEADS}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZephyraForQuestionAnswering(config).to(device)
    
    # Print model structure
    print("\nModel structure:")
    print(model)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, config.VOCAB_SIZE, (1, 512)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    start_positions = torch.tensor([0]).to(device)
    end_positions = torch.tensor([5]).to(device)
    
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        print("Forward pass successful!")
        print(f"Output keys: {outputs.keys()}")
        print(f"Loss: {outputs['loss'].item()}")
        print(f"Start logits shape: {outputs['start_logits'].shape}")
        print(f"End logits shape: {outputs['end_logits'].shape}")
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
    
    print("\nModel inspection complete.")

if __name__ == "__main__":
    inspect_model()