import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from src.model.zephyra import ZephyraForQuestionAnswering
from src.tokeniser.zephyratokeniser import ZephyraTokenizer
from src.utils.trainingUtils import train, evaluate, save_checkpoint, load_checkpoint, save_best_model
import src.config as config

print("\nImporting modules: [Complete]\n")

class CoQADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = torch.load(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']
        
        # Tokenize input
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get start and end positions
        answer_start = item['answer_start']
        answer_end = answer_start + len(answer)

        # Convert answer positions to token positions
        char_to_token = inputs.char_to_token(0)  # 0 for the first sequence
        if answer_start == -1 or answer_end == -1:
            start_position = end_position = 0  # set to [CLS] token for impossible answers
        else:
            start_position = char_to_token(answer_start)
            end_position = char_to_token(answer_end - 1)

            # If the answer is truncated, set positions to 0
            if start_position is None:
                start_position = end_position = 0
            elif end_position is None:
                end_position = start_position

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }
    

def load_preprocessed_data(file_path):
    data = torch.load(file_path)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    start_positions = data['start_positions']
    end_positions = data['end_positions']
    return TensorDataset(input_ids, attention_mask, start_positions, end_positions)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice set to: [{device}]\n")

    # Initialize tokenizer
    tokenizer = ZephyraTokenizer()
    print("\nTokenizer initialized: [Complete]\n")

    # Update vocabulary size in config
    config.VOCAB_SIZE = tokenizer.getVocabSize()
    print(f"\nVocabulary size: [{config.VOCAB_SIZE}]\n")

    # Initialize model
    model = ZephyraForQuestionAnswering(config).to(device)
    print("\nModel initialized and moved to device: [Complete]\n")

    # Load preprocessed datasets
    train_dataset = load_preprocessed_data(config.TRAIN_DATA_PATH)
    val_dataset = load_preprocessed_data(config.VAL_DATA_PATH)
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