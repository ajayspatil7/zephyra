
# config.py

# Model parameters
VOCAB_SIZE = 20000
D_MODEL = 768
NUM_LAYERS = 12
NUM_HEADS = 12
D_FF = 3072
MAX_SEQ_LENGTH = 1024
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WARMUP_STEPS = 1000
MAX_GRAD_NORM = 1.0

# Data parameters
TRAIN_DATA_PATH = "/Users/ajay/Downloads/zephyra/data/dataSet.txt"
VAL_DATA_PATH = "/Users/ajay/Downloads/zephyra/data/validation.txt"
TOKENIZER_PATH = "tokenizer.json" 

# Logging and saving
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = "checkpoints"
