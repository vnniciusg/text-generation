# Shakespeare Text Generator

A deep learning model that generates text in the style of Shakespeare's works using TensorFlow.

## Overview

This project implements a recurrent neural network (RNN) model to generate Shakespearean-style text. The model is trained on Shakespeare's writings to learn the patterns, vocabulary, and style characteristics of his works, and can then generate new text that mimics his distinctive writing style.

## Features

- Downloads and preprocesses Shakespeare's text data
- Implements a configurable RNN model (LSTM or GRU)
- Provides training functionality with checkpoints and TensorBoard visualization
- Includes a text generation interface with temperature control

## Project Structure

```
text-generation/
├── config/   # Configuration settings
├── data/     # Data loading and preprocessing
├── logs/     # TensorBoard logs
├── models/   # Model architecture, training, and generation
├── utils/    # Utility functions
├── main.py   # Main execution script
└── README.md # Project documentation
```

## Requirements

- Python 3.11+
- TensorFlow
- Pydantic
- Loguru

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vnniciusg/text-generation.git
   cd text-generation
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

Run the model training and text generation:

```bash
python main.py
```

This will:

- Download Shakespeare's text corpus
- Preprocess the data
- Train the model (with checkpoints saved and TensorBoard logging)
- Generate a sample text starting with "ROMEO: "

## Configuration

Model and training parameters can be customized in the `config/__init__.py` file:

- Data parameters (sequence length, batch size)
- Model architecture (LSTM or GRU, embedding dimensions, RNN units)
- Training parameters (epochs, learning rate)
- Generation parameters (temperature, output length)

## Text Generation

The model can generate text with different "temperature" values:

- Lower temperature (e.g., 0.2) produces more predictable text
- Higher temperature (e.g., 1.0) produces more creative/random text

## Performance Monitoring

All key functions are decorated with a `@timeit` decorator to measure execution time. The project also integrates TensorBoard for training visualization.

## License

[MIT License](LICENSE)
