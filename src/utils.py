import torch

def generate_text(model, tokenizer, start_string, seq_len, num_generate=100, temperature=1.0):
    model.eval()
    input_eval = torch.tensor(tokenizer.encode(start_string)).unsqueeze(0).to(next(model.parameters()).device)
    generated_text = start_string

    with torch.no_grad():
        for _ in range(num_generate):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :] / temperature
            predicted_id = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)
            
            input_eval = torch.cat([input_eval, predicted_id], dim=1)
            generated_text += tokenizer.decode([predicted_id.item()])

            if len(input_eval[0]) >= seq_len:
                input_eval = input_eval[:, 1:]

    return generated_text

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

"""<-----------End of utils.py----------->"""