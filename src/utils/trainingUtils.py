import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from src import config
import os

def train(model, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=config.USE_MIXED_PRECISION):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs['loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    return avg_loss

def evaluate(model, dataloader, device, epoch, writer):
    model.eval()
    total_loss = 0
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs['loss']
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']

            total_loss += loss.item()

            all_start_preds.extend(torch.argmax(start_logits, dim=1).tolist())
            all_end_preds.extend(torch.argmax(end_logits, dim=1).tolist())
            all_start_labels.extend(start_positions.tolist())
            all_end_labels.extend(end_positions.tolist())

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Validation/Loss', avg_loss, epoch)

    exact_match = calculate_exact_match(all_start_preds, all_end_preds, all_start_labels, all_end_labels)
    f1_score = calculate_f1_score(all_start_preds, all_end_preds, all_start_labels, all_end_labels)

    writer.add_scalar('Validation/ExactMatch', exact_match, epoch)
    writer.add_scalar('Validation/F1Score', f1_score, epoch)

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

def save_best_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Best model saved to {filename}")

def decode_predictions(tokenizer, context, start_pred, end_pred):
    return tokenizer.decode(context[start_pred:end_pred+1])