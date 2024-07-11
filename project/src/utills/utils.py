import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, dataloader, optimizer, scheduler, criterion, max_grad_norm, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item(), "lr": get_lr(optimizer)})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            batch = batch.to(device)
            
            outputs = model(batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)

def create_optimizer_and_scheduler(model, lr, warmup_steps, num_training_steps):
    optimizer = AdamW(model.parameters(), lr=lr)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler
