# This file contains the training arguments for the language model

VOCAB_SIZE = 0
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_ATTENTION_HEADS = 12
INTERMEDIATE_SIZE = 3072
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 15
TRAIN_DATA_PATH = "./data/train.txt"
VAL_DATA_PATH = "./data/validate.txt"
