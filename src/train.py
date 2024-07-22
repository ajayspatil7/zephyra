import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.zephyra import ZephyraModel
from utils.dataU import ZephyraCoQADataset
from tokenizer.tokenizer import ZephyraTokenizer
from utils.trainArgs import train_epoch, validate_epoch
import config
import os
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    print("\nModules loaded correctly\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to : [{device}]\n")

    tokenizer = ZephyraTokenizer()
    print(f"Tokenizer loaded on : [{device}]\n")

    config.VOCAB_SIZE = tokenizer.getVocabSize()
    print(f"Vocabulary size: [{config.VOCAB_SIZE}]\n")
    
    model = ZephyraModel(
        vocab_size=config.VOCAB_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE
    ).to(device)
    print(f"Model loaded on : [{device}] \n")

    train_dataset = ZephyraCoQADataset(config.TRAIN_DATA_PATH, tokenizer, config.MAX_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=ZephyraCoQADataset.collate_fn)
    
    val_dataset = ZephyraCoQADataset(config.VAL_DATA_PATH, tokenizer, config.MAX_SEQ_LENGTH)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=ZephyraCoQADataset.collate_fn)
    
    print(f"Dataset and Dataloader loaded on : [{device}] \n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE)
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    print(f"Optimizer and scheduler loaded, training started on : [{device}]\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        current_lr = get_lr(optimizer)
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Learning Rate: {current_lr:.6f}")

        train_loss = train_epoch(model, train_dataloader, optimizer, device, writer, epoch)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            val_loss = validate_epoch(model, val_dataloader, device, writer, epoch)
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Validation Loss: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
                print("New best model saved!")
            else:
                patience_counter += 1
                
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss if (epoch + 1) % config.VALIDATION_INTERVAL == 0 else None,
        }, os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt"))
        
    writer.close()
    print(f"Training complete and model saved\n")

if __name__ == "__main__":
    main()