class config:
    BATCH_SIZE_TRAIN = 16
    BATCH_SIZE_VALIDATION = 8
    BATCH_SIZE_TEST = 8
    DROPOUT = 0.2
    EPOCHS = 50
    FOLDS = 10
    HIDDEN_DIM = 64
    LAYER_DIM = 3
    LEARNING_RATE = 1e-3
    LR_FACTOR = 0.4  # BY HOW MUCH THE LR IS DECREASING
    LR_PATIENCE = 1  # 1 MODEL NOT IMPROVING UNTIL LR IS DECREASING
    OUTPUT_DIM = 1
    WEIGHT_DECAY = 1e-6