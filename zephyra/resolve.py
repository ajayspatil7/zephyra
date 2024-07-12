from pathlib import Path
from .utills.utils import load_model, load_tokenizer, generate_text
from .utills.config import CHECKPOINT_DIR, TOKENIZER_PATH

def main():
    model_path = f"{CHECKPOINT_DIR}/zephyra.pth"
    
    # Load the model and tokenizer
    model, device = load_model(model_path)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # Test the model
    prompt = "When should you watch out for vampires?"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()