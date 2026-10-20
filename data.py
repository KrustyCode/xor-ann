"""Toy dataset: the XOR truth table."""

import numpy as np

# Inputs, shape (4, 2): every combination of two binary features
X = np.asarray([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
], dtype=float)

# Targets, shape (4,): 1 when exactly one input is 1 (XOR)
y = np.asarray([0, 1, 1, 0 ], dtype=int)
