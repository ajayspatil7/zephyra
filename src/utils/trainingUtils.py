import torch
from torch.cuda.amp import GradScaler, autocast
from src import config
import torch.nn.functional as F
from src.model.zephyra import ZephyraModel
from tokenizer.tokenizer import ZephyraTokenizer


def trainEpoch(model, dataloader, optimizer, device, writer, epoch):
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

def validateEpoch(model, dataloader, device, writer, epoch):
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

def loadBestModel(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = ZephyraModel(
        vocab_size=config.VOCAB_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE
    )
    
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move the model to the appropriate device
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model, device

def inference(model, tokenizer, input_text, device):
    # Tokenize the input
    inputs = tokenizer.encode(input_text, add_special_tokens=True)
    input_ids = torch.tensor([inputs]).to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted token ids
    predicted_ids = torch.argmax(outputs, dim=-1)
    
    # Decode the output
    predicted_text = tokenizer.decode(predicted_ids[0].tolist())
    
    return predicted_text

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