# To run this file on colab do the following,
# After you clone the repo on colab,
# !git clone https://github.com/ajayspatil7/zephyra.git

# Add the root dir to the PATH '/content/zephyra/'
# import sys
# sys.path.append('/content/zephyra/')

# Then change change your working dir to root folder
# %cd zephyra

# Then, uncomment these below lines and run 'python src/train.py' to start the training
# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)
# !pip install tiktoken tensorboardx


import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
from model.zephyra import ZephyraForSequenceClassification
from tokenizers import ZephyraTokenizer
from src.utils.dataUtils import ZephyraCoQADataset
from src.utils.trainingUtils import train_epoch, evaluate, save_checkpoint, load_checkpoint
import config
import os
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Initialize tokenizer
    tokenizer = ZephyraTokenizer()

    # Load datasets
    train_dataset = ZephyraCoQADataset(config.TRAIN_DATA_PATH, tokenizer, config.MAX_SEQ_LENGTH)
    val_dataset = ZephyraCoQADataset(config.VAL_DATA_PATH, tokenizer, config.MAX_SEQ_LENGTH)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = ZephyraForSequenceClassification(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE,
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        num_labels=config.NUM_LABELS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.WARMUP_STEPS, 
        num_training_steps=len(train_dataloader) * config.NUM_EPOCHS
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Load checkpoint if resuming training
    start_epoch = 0
    if config.RESUME_FROM_CHECKPOINT and os.path.exists(config.RESUME_CHECKPOINT_PATH):
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, config.RESUME_CHECKPOINT_PATH, device)
        logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        # Train
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            
            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log training progress
            if global_step % config.LOGGING_STEPS == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)

        avg_train_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Average train loss: {avg_train_loss:.4f}")
        writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)

        # Validate
        val_loss, val_accuracy, val_f1 = evaluate(model, val_dataloader, device)
        logger.info(f"Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        # Log validation results
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        writer.add_scalar('Validation/F1', val_f1, epoch)

        # Save checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")

        # Early stopping
        if epoch - start_epoch >= config.EARLY_STOPPING_PATIENCE:
            if val_loss > best_val_loss:
                logger.info(f"Validation loss hasn't improved for {config.EARLY_STOPPING_PATIENCE} epochs. Stopping training.")
                break

        # Remove old checkpoints if necessary
        if config.SAVE_TOTAL_LIMIT:
            checkpoints = sorted([f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_")])
            if len(checkpoints) > config.SAVE_TOTAL_LIMIT:
                os.remove(os.path.join(config.CHECKPOINT_DIR, checkpoints[0]))

    logger.info("Training complete!")
    writer.close()

if __name__ == "__main__":
    main()