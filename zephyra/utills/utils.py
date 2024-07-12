import torch
from pathlib import Path
from zephyra.model.zephyra_model import ZephyraResolve
from zephyra.tokenization.bytepairencoding import BPETokenizer
from zephyra.utills.config import MODEL_CONFIG, TOKENIZER_PATH


def load_tokenizer(tokenizer_path: Path):
    tokenizer = BPETokenizer()
    tokenizer.load(str(tokenizer_path))
    return tokenizer

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

def load_model(model_path: Path, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ZephyraResolve(**MODEL_CONFIG)
    state_dict = torch.load(model_path, map_location=device)
    
    # Resize RotaryEmbedding if necessary
    for key, value in state_dict.items():
        if 'rotary_emb.inv_freq' in key:
            loaded_dim = value.shape[0]
            current_dim = model.state_dict()[key].shape[0]
            if loaded_dim != current_dim:
                print(f"Resizing {key} from {current_dim} to {loaded_dim}")
                block_index = int(key.split('.')[1])
                model.blocks[block_index].rotary_emb.resize_inv_freq(loaded_dim)
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model, device

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature
        )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text