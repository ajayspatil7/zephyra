import logging
import sys
# Configuring the logger
logging.basicConfig(level=logging.INFO, format=('%(asctime)s - %(levelname)s - %(message)s'), datefmt='%Y-%m-%d %I:%M:%S %p', filename='./zephyra.log')
logging.info(f'(Step -1) : {sys.argv[0]} scripted started running')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import train
import time
from transformerblock import Zephyra
from tokeniser import create_tokenizer
from trainingSamplesBuilder import QADataset
import logging


logging.info('(Step 0)  : All dependencies import complete')
time.sleep(3.0)


def main():
    # Hyperparameters
    embed_dim = 128
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    seq_len = 128
    batch_size = 64
    epochs = 10
    lr = 0.001

    technical_logger.info("Hyperparameters: ", embed_dim, num_heads, ff_dim, num_layers)

    logging.info("(Step 1)  : Tokeniser Loaded...")
    # Load tokenizer
    tokenizer = create_tokenizer('./cleanData.txt')
    logging.info("(Step 2)  : Tokenisation Complete...")


    logging.info("(Step 3)  : Dataset builder Loaded...")
    # Load dataset and dataloader
    dataset = QADataset('./cleanData.txt', tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logging.info("(Step 4)  : Dataset builder Complete...")


    logging.info("(Step 5)  : Loading model")
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Zephyra(tokenizer.vocab_size, embed_dim, num_heads, ff_dim, num_layers).to(device)
    
    logging.info(f"(Step 6)  : Zephyra model loaded on {device}")


    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"(Step 7)  : Zephyra training started on a {device}")
    logging.info("----------------------------------------------------------------------------------------------------\n" )


    # Training loop
    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    

    # Test the model
    model.eval()
    test_questions = [
        "What happens to you if you eat watermelon seeds?",
        "Where did fortune cookies originate?",
        "Why do veins appear blue?",
    ]

    for question in test_questions:
        answer = model.generate_answer(question, tokenizer)
        print(f"Question: {question}")
        print(f"Generated Answer: {answer}\n")


if __name__ == "__main__":
    main()

    
    

"""<-----------End of zephyra.py----------->"""