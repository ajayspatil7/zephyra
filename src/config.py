# This file contains the training arguments for the language model

VOCAB_SIZE = 0
HIDDEN_SIZE = 512
NUM_LAYERS = 8
NUM_ATTENTION_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 15
TRAIN_DATA_PATH = "./data/dataset/coqa_train.json"
VAL_DATA_PATH = "./data/dataset/coqa_valx.json"
GRADIENT_ACCUMULATION_STEPS = 1
USE_MIXED_PRECISION = True

