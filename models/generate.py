from typing import Union, overload

import tensorflow as tf

from utils import timeit
from config import Config
from .model import ShakespeareTextGeneratorModel


class ShakespeareTextGenerator:

    def __init__(self, model: ShakespeareTextGeneratorModel, char2idx, idx2char, config: Config):
        self.model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.config = config

    @overload
    def generate(self, start_string: str, temperature: float) -> str: ...

    @overload
    def generate(self, start_string: str, temperature: int) -> str: ...

    @timeit
    def generate(self, start_string: str, temperature: Union[float, int]) -> str:

        inference_model = tf.keras.models.clone_model(self.model)
        inference_model.build(tf.TensorShape([1, None]))
        inference_model.load_weights()

        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []
        inference_model.reset_states()

        for _ in range(self.config.GENERATION_NUM_GENERATE):
            predictions = tf.squeeze(inference_model(input_eval), 0) / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0]

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])

        return start_string + "".join(text_generated)
