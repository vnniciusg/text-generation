from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout

from config import Config


class ShakespeareTextGeneratorModel(Model):

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.config.MODEL_EMBEDDING_DIM)
        self.rnn_layers = [self._build_rnn_layer() for _ in range(2)]
        self.dropout = Dropout(self.config.MODEL_DROPOUT)
        self.dense = Dense(self.vocab_size)

    def _build_rnn_layer(self):

        if self.config.MODEL_ARCHITECTURE == "LSTM":
            return LSTM(self.config.MODEL_RNN_UNITS, return_sequences=True, stateful=True)

        return GRU(self.config.MODEL_RNN_UNITS, return_sequences=True, stateful=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        for rnn in self.rnn_layers:
            x = rnn(x)
            x = self.dropout(x)
        return self.dense(x)
