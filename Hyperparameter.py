#define model parameters specific to T5
model_params = {
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 64,  # training batch size
    "VAL_BATCH_SIZE": 64,  # validation batch size
    "TRAIN_EPOCHS": 30,  # number of training epochs
    "VAL_EPOCHS": 30,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 1024,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 1024,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}