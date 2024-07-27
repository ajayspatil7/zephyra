import torch
from torch.cuda.amp import GradScaler, autocast
from src import config
import torch.nn.functional as F
from src.model.zephyra import ZephyraModel
from tokenizer.tokenizer import ZephyraTokenizer
import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast():
            outputs = model(**batch)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), accuracy, f1

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# Inference Usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = ZephyraTokenizer()
# input_text = "Your input text here"
# output = inference(model, tokenizer, input_text, device)
# print(f"Input: {input_text}")
# print(f"Output: {output}")

# Load model Usage
# best_model_path = "./checkpoints/best_model.pt"
# model, device = loadBestModel(best_model_path)
# print(f"Model loaded on: {device}")