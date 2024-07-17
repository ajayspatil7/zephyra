
# src/train.py
import torch
from torch.utils.data import DataLoader
from model.zephyra import ZephyraModel
from tokenizer.tokenizer import ZephyraTokenizer
from utils.dataU import TextDataset
from utils.trainArgs import train_epoch
import config

def main():
    print("\nModules loaded correctly \n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to : [{device}] \n")

    tokenizer = ZephyraTokenizer()
    print(f"Tokenizer loaded on [{device}] \n")

    config.VOCAB_SIZE = tokenizer.get_vocab_size()
    
    
    model = ZephyraModel(
        vocab_size=config.VOCAB_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE
    ).to(device)
    print(f"Model loaded on [{device}] \n")

    train_dataset = TextDataset(config.TRAIN_DATA_PATH, tokenizer, config.MAX_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print(f"Dataset and Dataloader loaded \n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Optimiser loaded and training started on [{device}] \n")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"zephyraWeights.pt")
        print(f"Training complete and model saved\n")

if __name__ == "__main__":
    main()
