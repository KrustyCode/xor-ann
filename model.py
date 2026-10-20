"""A minimal feed-forward neural network written from scratch with NumPy.

Each layer exposes ``forward`` (also callable directly) and ``backward``.
``forward`` caches whatever ``backward`` needs, so ``backward`` must be called
after a ``forward`` pass. Layers with parameters (``Linear``) also have a
``step`` method that applies a gradient descent update.
"""

from abc import ABC, abstractmethod
import numpy as np
from config import (
    LEARNING_RATE,
    ERROR, SEED
)

# Shared, seeded generator so weight initialization is reproducible
RNG = np.random.default_rng(seed = SEED)

class Model(ABC):
    """Interface every model must implement."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        """Train the model on data."""
        pass

    @abstractmethod
    def predict(self):
        """Return predictions for new inputs."""
        pass

class Sigmoid:
    """Sigmoid activation: ``1 / (1 + e^-x)``, squashes values into (0, 1)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out =  1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_out: np.ndarray|None = None) -> np.ndarray:
        """Multiply ``grad_out`` by the local derivative ``s * (1 - s)``.

        If ``grad_out`` is None, the local derivative alone is returned.
        """
        local = self.out * (1 - self.out)
        return local if grad_out is None else grad_out * local

    __call__ = forward

class Linear:
    """Fully connected layer: ``x @ W + b``.

    Weights and biases are initialized uniformly in [-1, 1).
    """

    def __init__(self, input_size: int, output_size: int):
        self.W = 2 * RNG.random(size=(input_size, output_size)) - 1
        self.b = 2 * RNG.random(size=output_size) - 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Store gradients for ``W`` and ``b``; return the gradient w.r.t. the input."""
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def step(self, learning_rate:float = LEARNING_RATE ) -> None:
        """Apply one gradient descent update using the stored gradients."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    __call__ = forward

class ReLU:
    """ReLU activation: ``max(0, x)``."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Pass the gradient through only where the input was positive."""
        return grad_out * self.mask

    __call__ = forward

class Binary_Cross_entropy():
    """Mean binary cross-entropy loss.

    Predictions are clipped to ``[eps, 1 - eps]`` to avoid ``log(0)``.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Return the loss between targets ``y`` and predicted probabilities ``y_hat``."""
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Return the gradient of the mean loss w.r.t. ``y_hat``."""
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        return (y_hat - y) / (y_hat * (1 - y_hat)) / len(y)

class ANN(Model):
    """Two-layer network for binary classification.

    Architecture: Linear -> ReLU -> Linear -> Sigmoid.
    """

    def __init__(self, input_size:int, hidden_size: int, output_size:int):
        self.layers = [
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
            Sigmoid()
        ]
        self.loss_fn = Binary_Cross_entropy()

    def forward(self, x: np.ndarray) -> np.ndarray|float:
        """Run ``x`` through every layer and return the output probabilities."""
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, grad: np.ndarray|None = None) -> np.ndarray:
        """Backpropagate ``grad`` from the output through every layer in reverse."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, learning_rate: float = LEARNING_RATE) -> None:
        """Update the parameters of every layer that has weights."""
        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.step(learning_rate)

    __call__ = forward

    def fit(self, X: np.ndarray, y:np.ndarray, max_epochs: int = 10,
            target_error: float = ERROR, learning_rate: float = LEARNING_RATE) -> "Model":
        """Train with full-batch gradient descent.

        Stops after ``max_epochs`` or once the loss is at most ``target_error``.
        Returns ``self`` so calls can be chained.
        """
        y = y.reshape(-1, 1)  # match the (n_samples, 1) shape of the output
        for epoch in range (max_epochs):
            prediction = self(X)
            loss = self.loss_fn(y, prediction)
            if loss <= target_error:
                break
            self.backward(self.loss_fn.backward(y, prediction))
            self.step(learning_rate)

        return self

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return 0/1 class labels: 1 where the output probability is >= ``threshold``."""
        prediction = self(x)
        return (prediction >= threshold).astype(int)
