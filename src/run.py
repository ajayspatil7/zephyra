import torch
from model.zephyra import ZephyraModel
from tokenizer.tokenizer import ZephyraTokenizer
from utils.trainingUtils import loadBestModel, inference

def main():
    best_model_path = "./checkpoints/best_model.pt"
    model, device = loadBestModel(best_model_path)
    print(f"Model loaded on: {device}")
    
    tokenizer = ZephyraTokenizer()
    
    while True:
        input_text = input("Enter your input (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        
        output = inference(model, tokenizer, input_text, device)
        print(f"Input: {input_text}")
        print(f"Output: {output}")
        print()

if __name__ == "__main__":
    main()