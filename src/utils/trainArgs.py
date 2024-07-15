
import torch
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Create attention mask
        attention_mask = (batch != dataloader.dataset.tokenizer.get_pad_token_id()).float()
        
        outputs = model(batch, attention_mask=attention_mask)
        
        # Only consider non-padded tokens for loss calculation
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch.view(-1), ignore_index=dataloader.dataset.tokenizer.get_pad_token_id())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
