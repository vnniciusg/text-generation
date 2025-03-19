import numpy as np
from loguru import logger
from tensorflow.keras.utils import get_file
from tensorflow.data import Dataset

from config import Config


class ShakespeareDataLoader:

    def __init__(self, config: Config):
        self.config = config
        self.text = self._load_raw_data()
        self.vocab, self.char2idx, self.idx2char = self._build_vocab()
        logger.info(f"the vocab size is: {len(self.vocab)}")

    def _load_raw_data(self) -> str:
        path_to_file = get_file("shakespeare.txt", self.config.DATA_URL)
        return open(path_to_file, 'rb').read().decode(encoding='utf-8')

    def _build_vocab(self):
        vocab = sorted(set(self.text))
        return vocab, {c: i for i, c in enumerate(vocab)}, np.array(vocab)

    def _create_sequences(self):
        text_as_int = np.array([self.char2idx[c] for c in self.text])
        return Dataset.from_tensor_slices(text_as_int).batch(self.config.DATA_SEQ_LENGTH + 1, drop_remainder=True)

    def get_dataset(self):
        sequences = self._create_sequences()
        dataset = sequences.map(self._split_input_target)
        return dataset.shuffle(self.config.DATA_BUFFER_SIZE).batch(self.config.DATA_BATCH_SIZE, drop_remainder=True)
    
    def _split_input_target(self, chunk):
        return chunk[:-1], chunk[1:]
