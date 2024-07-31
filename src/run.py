import sys
import os
import warnings
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')

import torch
import re
from src.model.zephyra import ZephyraForQuestionAnswering
from tokeniser import ZephyraTokeniser, ZephyraTokens
from src.config import config


def load_model(model_path):
    model = ZephyraForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(config.DEVICE)))
    model.to(config.DEVICE)
    model.eval()
    return model

def get_answer(model, tokenizer, context, question):
    inputs = tokenizer.encode_plus(
        question,
        context,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True
    )
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs['start_logits'])
    answer_end = torch.argmax(outputs['end_logits']) + 1

    input_ids = inputs["input_ids"].squeeze().tolist()
    answer = tokenizer.decode(input_ids[answer_start:answer_end])
    return answer

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    model_path = "./checkpoints/best_model.pt"
    vocab_path = "./data/datasets/vocab.txt"
    
    vocab = load_vocab(vocab_path)
    tokenizer = ZephyraTokeniser(vocab, ZephyraTokens)
    tokenizer.print_vocab_info()
    model = load_model(model_path)

    print(f"Zephyra Model Loaded on cuda. Ready for questions!")
    print("Enter 'quit' to exit.")

    while True:
        context = input("\nEnter context : ")
        if context.lower() == 'quit':
            break

        question = input("\nEnter question : ")
        if question.lower() == 'quit':
            break
        
        print("Processing...")
        answer = get_answer(model, tokenizer, context, question)
        print(f"\nAnswer: {answer}")
        

if __name__ == "__main__":
    main()