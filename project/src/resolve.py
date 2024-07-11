
from utills import load_model, generate_text
from tokenization.bytepairencoding import BPETokenizer

def main():
    model_path = "/Users/ajay/Downloads/zephyra/project/src/utills/zephyra.pth"
    tokenizer_path = "/Users/ajay/Downloads/zephyra/project/src/tokenizer.json"

    # Load the model and tokenizer
    model, device = load_model(model_path, tokenizer_path)
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Test the model
    prompt = "When should you watch out for vampires?"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()