import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from src import config
import os
import json
import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def train(model, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        try:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            
            loss = outputs['loss']
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item() if loss is not None else 'N/A':.4f}")

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            print("Batch content:")
            for key, value in batch.items():
                print(f"  {key}: {type(value)}, Shape: {value.shape}")
            print("Model output shapes:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Shape: {value.shape}")
            raise

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Training/Loss', avg_loss, epoch)
    return avg_loss


def evaluate(model, dataloader, device, epoch, writer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}"):
            input_ids, attention_mask, start_positions, end_positions = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    return avg_loss


def calculate_exact_match(start_preds, end_preds, start_labels, end_labels):
    correct = sum((s_pred == s_label) and (e_pred == e_label) 
                  for s_pred, e_pred, s_label, e_label in zip(start_preds, end_preds, start_labels, end_labels))
    return correct / len(start_preds)

def calculate_f1_score(start_preds, end_preds, start_labels, end_labels):
    f1_scores = []
    for s_pred, e_pred, s_label, e_label in zip(start_preds, end_preds, start_labels, end_labels):
        pred_range = set(range(s_pred, e_pred + 1))
        label_range = set(range(s_label, e_label + 1))
        
        if len(pred_range) == 0 and len(label_range) == 0:
            f1_scores.append(1.0)
        elif len(pred_range) == 0 or len(label_range) == 0:
            f1_scores.append(0.0)
        else:
            intersection = len(pred_range.intersection(label_range))
            precision = intersection / len(pred_range)
            recall = intersection / len(label_range)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores)

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filename}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filename}")
        return 0, None

def save_best_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Best model saved to {filename}")

def decode_predictions(tokenizer, context, start_pred, end_pred):
    return tokenizer.decode(context[start_pred:end_pred+1])


def collateData(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    start_positions = torch.tensor([item['start_positions'] for item in batch])
    end_positions = torch.tensor([item['end_positions'] for item in batch])

    # Pad sequences to the maximum length in this batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'start_positions': start_positions,
        'end_positions': end_positions
    }


class CoQADataset(Dataset):
    def __init__(self, data, max_length=512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine question and context
        input_ids = item['question'] + item['context']
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Pad if necessary
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length  # Assuming 0 is the padding token ID
        attention_mask += [0] * padding_length

        # Adjust answer positions
        start_position = min(item['rationale_start'] + len(item['question']), self.max_length - 1)
        end_position = min(item['rationale_end'] + len(item['question']), self.max_length - 1)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

def load_preprocessed_data(file_path):
    print(f"Loading data from {file_path}")
    data = torch.load(file_path)
    return CoQADataset(data)
