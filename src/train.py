import sys
import os
import warnings
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)
# warnings.filterwarnings('ignore')

import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from src.model.zephyra import ZephyraForQuestionAnswering
# from src.tokeniser.zephyratokeniser import ZephyraTokenizer
from src.utils.trainingUtils import train, evaluate, save_checkpoint, load_checkpoint, save_best_model, CoQADataset


def main():
    # Configuration
    import src.config as config
    
    train_data = torch.load(config.TRAIN_INPUT_PATH)
    print(type(train_data))  # Should be dict
    print(train_data.keys())  # Should include 'inputs' and 'targets'

    val_data = torch.load(config.VAL_INPUT_PATH)
    print(type(val_data))  # Should be dict
    print(val_data.keys())  # Should include 'inputs' and 'targets'


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice set to: [{device}]\n")

    # Initialize model
    model = ZephyraForQuestionAnswering(config).to(device)
    print("\nModel initialized and moved to device: [Complete]\n")

    # Load preprocessed datasets
    train_dataset = CoQADataset(config.TRAIN_INPUT_PATH, config.TRAIN_TARGET_PATH, config.MAX_LEN)
    val_dataset = CoQADataset(config.VAL_INPUT_PATH, config.VAL_TARGET_PATH, config.MAX_LEN)
    print("\nPreprocessed datasets loaded: [Complete]\n")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print("\nDataLoaders created: [Complete]\n")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    print("\nOptimizer initialized: [Complete]\n")

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
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
        
        train_loss = train(model, train_dataloader, optimizer, device, epoch, writer)
        print(f"\nTraining for epoch {epoch+1} completed. Train Loss: [{train_loss:.4f}]\n")
        
        val_loss, exact_match, f1_score = evaluate(model, val_dataloader, device, epoch, writer)
        print(f"\nValidation for epoch {epoch+1} completed:")
        print(f"Validation Loss: [{val_loss:.4f}]")
        print(f"Exact Match: [{exact_match:.4f}]")
        print(f"F1 Score: [{f1_score:.4f}]\n")

        # Checkpoint saving
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        print(f"\nCheckpoint saved for epoch {epoch+1}\n")

        # Best model saving
        if f1_score > best_f1:
            best_f1 = f1_score
            save_best_model(model, best_model_path)
            print(f"\nNew best model saved with F1 Score: [{best_f1:.4f}]\n")

        # Learning rate scheduling
        scheduler.step(val_loss)
        print(f"\nLearning rate adjusted. Current LR: [{optimizer.param_groups[0]['lr']:.6f}]\n")

        # Early stopping check
        if optimizer.param_groups[0]['lr'] < config.MIN_LEARNING_RATE:
            print("\nLearning rate too small. Stopping training.\n")
            break

    print("\nTraining completed.\n")
    writer.close()
    print("\nTensorBoard writer closed.\n")

if __name__ == "__main__":
    main()
    print("\nScript execution completed.\n")
