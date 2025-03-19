from pydantic_settings import BaseSettings


class Config(BaseSettings):

    ENV: str = "dev"

    DATA_URL: str = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    DATA_SEQ_LENGTH: int = 100
    DATA_BUFFER_SIZE: int = 1000
    DATA_BATCH_SIZE: int = 64

    MODEL_ARCHITECTURE: str = "LSTM"
    MODEL_EMBEDDING_DIM: int = 256
    MODEL_RNN_UNITS: int = 1024
    MODEL_DROPOUT: float = 0.2

    TRAINING_EPOCHS: int = 30
    TRAINING_CHECKPOINT_DIR: str = "./saved_models/training_checkpoints"
    TRAINING_LEARNING_RATE: float = 1e-3

    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_NUM_GENERATE: int = 500
