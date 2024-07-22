import torch
from torch.cuda.amp import GradScaler, autocast
import config
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        # Create attention mask
        attention_mask = (batch != dataloader.dataset.tokenizer.get_pad_token_id()).float().unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        with autocast(enabled=config.USE_MIXED_PRECISION):
            outputs = model(batch, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch.view(-1), ignore_index=dataloader.dataset.tokenizer.get_pad_token_id())
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        
        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS

    return total_loss / len(dataloader)

# In the main function:
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
