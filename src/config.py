# config.py

class ZephyraConfig:
    # Model architecture
    VOCAB_SIZE = None  # Set this based on your tokenizer
    HIDDEN_SIZE = 512
    NUM_LAYERS = 8
    NUM_ATTENTION_HEADS = 8
    INTERMEDIATE_SIZE = 2048

    # Training args
    MAX_SEQ_LENGTH = 256
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 5
    TRAIN_DATA_PATH = "./data/raw/coqa-train-v1.0.json"
    VAL_DATA_PATH = "./data/raw/coqa-dev-v1.0.json"
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_MIXED_PRECISION = True

    # Validation args
    VALIDATION_INTERVAL = 1
    PATIENCE = 3
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER_FACTOR = 0.1
    LR_SCHEDULER_PATIENCE = 2

    # Tokenizer settings
    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    SEP_TOKEN_ID = 3

    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"ZephyraConfig has no attribute '{key}'")
