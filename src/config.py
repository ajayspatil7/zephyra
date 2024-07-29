import os
import torch

# Data paths
TRAIN_DATA_PATH = './data/dataset/tokenized_train_data.pt'
VAL_DATA_PATH = './data/dataset/tokenized_dev_data.pt'
TOKENIZER_PATH = './data/dataset/zephyra_tokenizer.pt'

# Model parameters
VOCAB_SIZE = None  # This will be set dynamically based on the tokenizer
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
MAX_SEQ_LENGTH = 512
INTERMEDIATE_SIZE = 3072
MAX_POSITION_EMBEDDINGS = 512
HIDDEN_ACT = "gelu"
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1
LAYER_NORM_EPS = 1e-12

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 15
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 0

# Optimizer parameters
ADAM_EPSILON = 1e-8
MIN_LEARNING_RATE = 1e-6

# Early stopping
PATIENCE = 3
MIN_DELTA = 0.001

# Mixed precision training
USE_MIXED_PRECISION = True

# Logging and saving
LOG_DIR = './logs'
CHECKPOINT_DIR = './checkpoints'
LOGGING_STEPS = 100
SAVE_STEPS = 1000

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tokenizer
TOKENIZER_MODEL = "cl100k_base"

# Evaluation
EVAL_BATCH_SIZE = 16

# Data processing
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30

# Random seed for reproducibility
SEED = 42

# TensorBoard
TENSORBOARD_UPDATE_FREQ = 100

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_PATIENCE = 2

# Checkpoint settings
SAVE_TOTAL_LIMIT = 5

# Model version
MODEL_VERSION = "v1.0.0"

# Debugging
DEBUG = False

# Distributed training
WORLD_SIZE = 1
DISTRIBUTED = False

# Additional parameters
TYPE_VOCAB_SIZE = 2
PAD_TOKEN_ID = 0
INITIALIZER_RANGE = 0.02
USE_CACHE = True
OUTPUT_HIDDEN_STATES = False
USE_RETURN_DICT = True

# Create a configuration class
class ZephyraConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Create the config object
config = ZephyraConfig(**{k: v for k, v in globals().items() if not k.startswith('__') and k.isupper()})
