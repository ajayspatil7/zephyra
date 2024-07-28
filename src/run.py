import sys
import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

import torch
from model.zephyra import ZephyraModel
from tokenizers.tokenizer import ZephyraTokenizer

def load_and_test_model(model_path):
    model = ZephyraModel(vocab_size=100277, hidden_size=512, num_layers=8, num_attention_heads=8, intermediate_size=2048)

    # Initialize tokenizer
    tokenizer = ZephyraTokenizer()

    # Prepare a sample input
    input_text = "Hey Zephyra"
    input_ids = torch.tensor([tokenizer.encode(input_text)]).long()

    # Run inference
    with torch.no_grad():
        output = model(input_ids)

    # Process output
    predicted_token_ids = torch.argmax(output, dim=-1)
    predicted_text = tokenizer.decode(predicted_token_ids[0].tolist())

    print(f"Input: {input_text}")
    print(f"Output: {predicted_text}")


def check_tokenizer():
    tokenizer = ZephyraTokenizer()
    
    sample_text = "Who are you?"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original text: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {tokenizer.getVocabSize()}")



if __name__ == "__main__":
    model_path = "/Users/ajay/Downloads/zephyra/checkpoints/bestmodel.pt"
    load_and_test_model(model_path)
    # check_tokenizer()