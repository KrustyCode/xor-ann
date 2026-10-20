"""Hyperparameters for the network and training loop."""

INPUT_NODES = 2         # number of input features
OUTPUT_NODES = 1        # number of output units
HIDDEN_NODES = 5        # number of hidden units
LEARNING_RATE = 0.04    # gradient descent step size
ERROR = 0.0004          # stop training once the loss drops to this value
SEED = 42               # RNG seed for reproducible weight initialization
