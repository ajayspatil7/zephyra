class Config:
    # Data paths
    TRAIN_PATH = "./data/datasets/train.pt"
    VAL_PATH = "./data/datasets/dev.pt"

    # Model parameters
    VOCAB_SIZE = 24710
    HIDDEN_SIZE = 1024
    NUM_HIDDEN_LAYERS = 16
    NUM_ATTENTION_HEADS = 16
    INTERMEDIATE_SIZE = 4096
    MAX_POSITION_EMBEDDINGS = 512
    MAX_LEN = 512
    HIDDEN_ACT = "gelu"
    HIDDEN_DROPOUT_PROB = 0.1
    ATTENTION_PROBS_DROPOUT_PROB = 0.1
    LAYER_NORM_EPS = 1e-12

    # Special token IDs
    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    WARMUP_STEPS = 0

    # Optimizer parameters
    ADAM_EPSILON = 1e-8
    MIN_LEARNING_RATE = 1e-8

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

    # Device
    DEVICE = 'cuda'  # or 'cpu' depending on availability

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

    # Additional model parameters
    TYPE_VOCAB_SIZE = 2
    INITIALIZER_RANGE = 0.02
    USE_CACHE = True
    OUTPUT_HIDDEN_STATES = False
    USE_RETURN_DICT = True

config = Config()