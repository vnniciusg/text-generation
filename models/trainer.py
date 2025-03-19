import os
from datetime import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from utils import timeit
from config import Config
from .model import ShakespeareTextGeneratorModel


class ShakespeareTextGeneratorTrainer:

    def __init__(self, model: ShakespeareTextGeneratorModel, dataset, config: Config) -> None:
        self.config = config
        self.model = model
        self.dataset = dataset
        self.optimizer = Adam(self.config.TRAINING_LEARNING_RATE)
        self.checkpoint_callback = ModelCheckpoint(filepath=os.path.join(self.config.TRAINING_CHECKPOINT_DIR, 'ckpt_{epoch}.weights.h5'), save_weights_only=True)
        self.tensorboard_callback = TensorBoard(log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    @timeit
    def train(self):

        self.model.compile(optimizer=self.optimizer, loss=SparseCategoricalCrossentropy(from_logits=True))

        return self.model.fit(self.dataset, epochs=self.config.TRAINING_EPOCHS, callbacks=[self.checkpoint_callback, self.tensorboard_callback])
