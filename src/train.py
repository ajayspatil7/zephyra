import sys
import os
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from src.model.zephyra import ZephyraForQuestionAnswering
import src.config as config
from src.utils.trainingUtils import save_checkpoint, load_checkpoint, save_best_model, train, evaluate, load_preprocessed_data
from src.tokenizers.tokenizer import ZephyraTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice set to: [{device}]\n")

    # Load preprocessed datasets
    try:
        tokenizer = ZephyraTokenizer()  # Load your tokenizer
        train_dataset = load_preprocessed_data(config.TRAIN_DATA_PATH)
        val_dataset = load_preprocessed_data(config.VAL_DATA_PATH)
        print("\nPreprocessed datasets loaded: [Complete]\n")
    except Exception as e:
        print(f"\nError loading preprocessed datasets: {e}\n")
        return

    # Load tokenizer
    try:
        tokenizer = torch.load(config.TOKENIZER_PATH)
        config.VOCAB_SIZE = tokenizer.getVocabSize()
        print(f"\nVocabulary size: [{config.VOCAB_SIZE}]\n")
    except Exception as e:
        print(f"\nError loading tokenizer: {e}\n")
        return

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print("\nDataLoaders created: [Complete]\n")

    # Initialize model
    model = ZephyraForQuestionAnswering(config).to(device)
    print(f"\nModel initialized and moved to : [{device}]\n")

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
    best_loss = float('inf')
    if os.path.exists(checkpoint_path):
        start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"\nCheckpoint loaded. Resuming from epoch: [{start_epoch}]\n")
    else:
        print("\nNo checkpoint found. Starting training from scratch.\n")

    print("\nStarting training loop\n")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{config.NUM_EPOCHS} {'='*20}\n")
        
        try:
            train_loss = train(model, train_dataloader, optimizer, device, epoch, writer)
            print(f"\nTraining for epoch {epoch+1} completed. Train Loss: [{train_loss:.4f}]\n")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            break
        
        val_loss = evaluate(model, val_dataloader, device, epoch, writer)
        print(f"\nValidation for epoch {epoch+1} completed. Validation Loss: [{val_loss:.4f}]\n")

        # Checkpoint saving
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        print(f"\nCheckpoint saved for epoch {epoch+1}\n")

        # Best model saving
        if val_loss < best_loss:
            best_loss = val_loss
            save_best_model(model, best_model_path)
            print(f"\nNew best model saved with Validation Loss: [{best_loss:.4f}]\n")

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

def check_data_format(file_path):
    data = torch.load(file_path)
    print(type(data))
    print(len(data.keys()) if isinstance(data, dict) else f"printing data")
    
    
    
if __name__ == "__main__":
#     check_data_format("./data/dataset/tokenized_train_data.pt")
    main()
    print("\nScript execution completed.\n")
