import torch
from torch.cuda.amp import GradScaler, autocast
import config
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(enabled=config.USE_MIXED_PRECISION):
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Reshape outputs and labels
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            # Print shapes for debugging
            # print(f"Batch {i} - Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            
            loss = F.cross_entropy(outputs, labels, ignore_index=dataloader.dataset.get_pad_token_id())
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        
        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        
        # Log training loss
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(dataloader) + i)

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device, writer, epoch):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Reshape outputs and labels
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            # Print shapes for debugging
            # print(f"Validation Batch {i} - Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            
            loss = F.cross_entropy(outputs, labels, ignore_index=dataloader.dataset.get_pad_token_id())

            total_loss += loss.item()
            
            # Log validation loss
            writer.add_scalar('Validation/Loss', loss.item(), epoch * len(dataloader) + i)

    return total_loss / len(dataloader)