<h1> XOR-ANN </h1>

A small feed-forward neural network built from scratch with NumPy, trained on XOR.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/KrustyCode/xor-ann.git
```

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Prints the predicted class (0 or 1) for each XOR input.

## Files

| File              | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `config.py`       | Layer sizes, learning rate, stopping loss, random seed      |
| `data.py`         | XOR dataset (`X`, `y`)                                      |
| `model.py`        | Layers (`Linear`, `ReLU`, `Sigmoid`), BCE loss, `ANN` model |
| `main.py`         | Builds, trains and evaluates the model                      |
| `requirements.py` | Python dependencies (pip install -r requirements.txt)       |

## Model

`Linear -> ReLU -> Linear -> Sigmoid`, trained with full-batch gradient descent
and binary cross-entropy loss.
