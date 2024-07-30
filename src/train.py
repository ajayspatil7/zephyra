import sys
import os
import warnings
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')
import wandb
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from src.model.zephyra import ZephyraForQuestionAnswering
from src.config import Config
from src.utils.trainingUtils import trainEpoch, evaluate, save_checkpoint, load_checkpoint, save_best_model, CoQADataset, pad_collate


warnings.filterwarnings('ignore')

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def main():
    config = Config()
    filtered_config = {k: v for k, v in vars(config).items() if is_jsonable(v)}
    wandb.init(project="zephyra-run-one", name="experiment-1", config=filtered_config)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nDevice set to: [{device}]\n")

    # Initialize model
    model = ZephyraForQuestionAnswering(config).to(device)
    print("\nModel initialized and moved to device: [Complete]\n")

    # Load preprocessed datasets
    train_dataset = CoQADataset(config.TRAIN_PATH, config.MAX_LEN, config.VOCAB_SIZE)
    val_dataset = CoQADataset(config.VAL_PATH, config.MAX_LEN, config.VOCAB_SIZE)
    print("\nPreprocessed datasets loaded: [Complete]\n")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    print("\nDataLoaders created: [Complete]\n")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, eps=config.ADAM_EPSILON)
    print("\nOptimizer initialized: [Complete]\n")

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.PATIENCE, verbose=True)
    print("\nLearning rate scheduler initialized: [Complete]\n")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    print(f"\nTensorBoard writer initialized. Logs will be saved to: [{config.LOG_DIR}]\n")

    # Set up checkpointing
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint.pt')
    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
    print(f"\nCheckpoint path set to: [{checkpoint_path}]\n")
    print(f"\nBest model path set to: [{best_model_path}]\n")

    # Load checkpoint if it exists
    start_epoch = 0
    best_f1 = 0
    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"\nCheckpoint loaded. Resuming from epoch: [{start_epoch}]\n")
    else:
        print("\nNo checkpoint found. Starting training from scratch.\n")

    print("\nStarting training loop\n")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{config.NUM_EPOCHS} {'='*20}\n")
        
        train_loss = trainEpoch(model, train_loader, optimizer, device, epoch, config)
        print(f"\nTraining for epoch {epoch+1} completed. Train Loss: [{train_loss:.4f}]\n")
        
        val_loss, exact_match, f1_score = evaluate(model, val_loader, device, epoch, config)
        
        wandb.log({
            "train/loss": train_loss,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/epoch": epoch,
            "val/loss": val_loss,
            "val/exact_match": exact_match,
            "val/f1_score": f1_score
        })
        
        print(f"\nValidation for epoch {epoch+1} completed:")
        print(f"Validation Loss: [{val_loss:.4f}]")
        print(f"Exact Match: [{exact_match:.4f}]")
        print(f"F1 Score: [{f1_score:.4f}]\n")

        # Checkpoint saving
        if epoch % config.SAVE_STEPS == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"\nCheckpoint saved for epoch {epoch+1}\n")

        # Best model saving
        if f1_score > best_f1:
            best_f1 = f1_score
            save_best_model(model, best_model_path)
            print(f"\nNew best model saved with F1 Score: [{best_f1:.4f}]\n")
            wandb.save(best_model_path)

        # Learning rate scheduling
        scheduler.step(val_loss)
        print(f"\nLearning rate adjusted. Current LR: [{optimizer.param_groups[0]['lr']:.6f}]\n")

        # Early stopping check
        if optimizer.param_groups[0]['lr'] < config.MIN_LEARNING_RATE:
            print("\nLearning rate too small. Stopping training.\n")
            break

    print("\nTraining completed.\n")
    writer.close()
    wandb.finish()
    print("\nTensorBoard writer closed and WandB run finished.\n")

if __name__ == "__main__":
    main()
    print("\nScript execution completed.\n")