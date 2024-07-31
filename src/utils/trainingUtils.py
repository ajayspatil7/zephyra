import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from src import config
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import wandb

class CoQADataset(Dataset):
    def __init__(self, data_path, max_len, vocab_size):
        data = torch.load(data_path)
        self.inputs = data['inputs']
        self.targets = data['targets']
        self.max_len = max_len
        self.vocab_size = vocab_size

        if not isinstance(self.inputs, list) or not isinstance(self.targets, list):
            raise TypeError("Inputs and targets should be lists.")
        
        if len(self.inputs) != len(self.targets):
            raise ValueError("Mismatch in length between inputs and targets.")

        print(f"Dataset loaded. Number of samples: {len(self.inputs)}")
        print(f"Sample input length: {len(self.inputs[0])}")
        print(f"Sample target: {self.targets[0]}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_item = self.inputs[idx][:self.max_len]
        target_item = self.targets[idx]
        
        # Convert to torch tensors
        input_item = torch.tensor(input_item, dtype=torch.long)
        
        # Ensure target_item has at least two elements (start and end position)
        if len(target_item) < 2:
            target_item = target_item + [0] * (2 - len(target_item))
        target_item = torch.tensor(target_item[:2], dtype=torch.long)

        # Clamp input values to be within vocab size
        input_item = torch.clamp(input_item, 0, self.vocab_size - 1)

        # Create attention mask
        attention_mask = torch.ones_like(input_item)

        return {
            'input_ids': input_item,
            'attention_mask': attention_mask,
            'target_ids': target_item
        }

    
def pad_collate(batch):
    inputs = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    targets = [item['target_ids'] for item in batch]

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_targets = torch.stack(targets)  # Changed from pad_sequence to stack

    return {
        'input_ids': padded_inputs,
        'attention_mask': padded_attention_masks,
        'target_ids': padded_targets
    }

def trainEpoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0
    scaler = GradScaler()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        
#         print(f"Batch {batch_idx}:")
#         print(f"input_ids shape: {batch['input_ids'].shape}")
#         print(f"attention_mask shape: {batch['attention_mask'].shape}")
#         print(f"target_ids shape: {batch['target_ids'].shape}")

        try:
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['target_ids'][:, 0],
                    end_positions=batch['target_ids'][:, 1],
                    return_dict=True
                )
                loss = outputs['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % config.LOGGING_STEPS == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch,
                    "train/step": epoch * len(dataloader) + batch_idx
                })

            print(f"Loss: {loss.item()}")

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    avg_loss = total_loss / len(dataloader)
    wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch})
    return avg_loss

# Update other functions similarly to accept config as a parameter

def evaluate(model, dataloader, device, epoch, config):
    model.eval()
    total_loss = 0
    all_start_preds, all_end_preds = [], []
    all_start_labels, all_end_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['target_ids'][:, 0],
                end_positions=batch['target_ids'][:, 1],
                return_dict=True
            )
            
            loss = outputs['loss']
            total_loss += loss.item()

            all_start_preds.extend(torch.argmax(outputs['start_logits'], dim=1).tolist())
            all_end_preds.extend(torch.argmax(outputs['end_logits'], dim=1).tolist())
            all_start_labels.extend(batch['target_ids'][:, 0].tolist())
            all_end_labels.extend(batch['target_ids'][:, 1].tolist())

    avg_loss = total_loss / len(dataloader)
    exact_match = calculate_exact_match(all_start_preds, all_end_preds, all_start_labels, all_end_labels)
    f1_score = calculate_f1_score(all_start_preds, all_end_preds, all_start_labels, all_end_labels)

    wandb.log({
        "val/loss": avg_loss,
        "val/exact_match": exact_match,
        "val/f1_score": f1_score,
        "val/epoch": epoch
    })

    return avg_loss, exact_match, f1_score


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
    
def load_model(model_path):
    model = ZephyraForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(config.DEVICE)))
    model.to(config.DEVICE)
    model.eval()
    return model

def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def get_answer(model, tokenizer, context, question):
    inputs = tokenizer.encode_plus(
        question,
        context,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True
    )
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    print(f"Input shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs['start_logits'][0]
    end_logits = outputs['end_logits'][0]

    # Find the tokens with the highest start and end logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Ensure end_index is not before start_index
    if end_index < start_index:
        end_index = torch.argmax(end_logits[start_index:]) + start_index

    print(f"Start index: {start_index}, End index: {end_index}")

    input_ids = inputs["input_ids"].squeeze().tolist()

    # Extract answer tokens
    answer_tokens = input_ids[start_index:end_index+1]
    
    # Decode tokens
    answer = tokenizer.decode(answer_tokens)

    return answer.strip()

def save_best_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Best model saved to {filename}")

def decode_predictions(tokenizer, context, start_pred, end_pred):
    return tokenizer.decode(context[start_pred:end_pred+1])


def main():
    train = CoQADataset()


if __name__ == "__main__":
    main()
    print("\nScript execution completed.\n")