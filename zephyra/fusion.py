# fusion.py

import torch
import torch.nn as nn
import os
from tqdm import tqdm

from resolve import ZephyraResolve
from utills.dataset import create_dataloaders
from utills.utils import train, evaluate, create_optimizer_and_scheduler
from utills import config

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
model = ZephyraResolve(
    vocab_size=config.VOCAB_SIZE,
    d_model=config.D_MODEL,
    num_layers=config.NUM_LAYERS,
    num_heads=config.NUM_HEADS,
    d_ff=config.D_FF,
    max_seq_length=config.MAX_SEQ_LENGTH,
    dropout=config.DROPOUT,
    tokenizer_path=config.TOKENIZER_PATH
)
model.to(device)

# Create dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    config.TRAIN_DATA_PATH,
    config.VAL_DATA_PATH,
    config.TOKENIZER_PATH,
    config.BATCH_SIZE,
    config.MAX_SEQ_LENGTH
)

# Set up optimizer and scheduler
num_training_steps = len(train_dataloader) * config.NUM_EPOCHS
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    config.LEARNING_RATE,
    config.WARMUP_STEPS,
    num_training_steps
)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
for epoch in range(config.NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
    
    train_loss = train(model, train_dataloader, optimizer, scheduler, criterion, config.MAX_GRAD_NORM, device)
    val_loss = evaluate(model, val_dataloader, criterion, device)
    
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % config.SAVE_INTERVAL == 0:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

print("Training complete!")