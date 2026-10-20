"""Entry point: train the ANN on the XOR dataset and print its predictions."""

from config import (INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
from data import (X, y)
from model import ANN

my_model = ANN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
my_model = my_model.fit(X, y, max_epochs=150)


n_data = X.shape[0]
prediction = my_model.predict(X).ravel()
right = 0

for i in range (n_data):
    if prediction[i] == y[i]:
        right +=1

accuracy = right/n_data

print(f"Accuracy: {accuracy:.2f}")
print(f"Prediction: {prediction}")
print(f"True Label: {y}")
