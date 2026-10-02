"""
A small feed-forward neural network, implemented from scratch with NumPy.

Architecture (deliberately simple, to stay easy to explain and fast on a
laptop CPU):

    Input layer:   784 units  (one per pixel of a flattened 28x28 digit)
    Hidden layer:   64 units, ReLU activation
    Output layer:   10 units, softmax activation (one score per digit 0-9)

    X (784) --W1,b1--> Z1 --ReLU--> A1 (64) --W2,b2--> Z2 --softmax--> A2 (10)

Every neuron in one layer is connected to every neuron in the next
("fully connected" / "dense"). W1, b1, W2, b2 are the learnable parameters:
- W1: (784, 64) weight matrix between input and hidden layer
- b1: (64,)     bias added to each hidden neuron
- W2: (64, 10)  weight matrix between hidden and output layer
- b2: (10,)     bias added to each output neuron
"""

import numpy as np

from .activations import relu, relu_derivative, softmax
from .losses import accuracy, cross_entropy_loss


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=64, output_size=10, seed=42):
        rng = np.random.default_rng(seed)

        # He initialization: random small weights scaled by sqrt(2 / fan_in).
        # This keeps the signal from exploding or vanishing as it passes
        # through layers, which matters a lot with ReLU activations.
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        """
        Forward propagation: pushes a batch of images through the network
        to get class probabilities.

        X: (batch_size, 784) input pixels.
        Returns the output probabilities A2: (batch_size, 10), plus a cache
        of intermediate values needed by backward().
        """
        Z1 = X @ self.W1 + self.b1        # weighted sum into the hidden layer
        A1 = relu(Z1)                     # hidden layer activations
        Z2 = A1 @ self.W2 + self.b2        # weighted sum into the output layer
        A2 = softmax(Z2)                  # output probabilities (sum to 1 per row)

        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def backward(self, y_true, cache):
        """
        Backpropagation: works out how much each weight and bias contributed
        to the error, by applying the chain rule layer by layer, starting
        from the output and moving backward to the input.

        y_true: (batch_size, 10) one-hot encoded true labels.
        Returns a dict of gradients, one per parameter.
        """
        X, A1, Z1, A2 = cache["X"], cache["A1"], cache["Z1"], cache["A2"]
        num_samples = X.shape[0]

        # Gradient of cross-entropy loss w.r.t. the output layer's raw
        # scores (Z2) collapses to this simple, well-known expression when
        # softmax and cross-entropy are combined: predicted - actual.
        dZ2 = (A2 - y_true) / num_samples

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        # Push the error backward through W2 into the hidden layer...
        dA1 = dZ2 @ self.W2.T
        # ...then through ReLU: error only flows where the neuron was active.
        dZ1 = dA1 * relu_derivative(Z1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_params(self, grads, learning_rate):
        """Gradient descent step: move every parameter a small distance
        (learning_rate) in the direction that reduces the loss."""
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]

    def predict(self, X):
        """Returns the predicted digit (0-9) for each input row."""
        probabilities, _ = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        batch_size=64,
        learning_rate=0.1,
        verbose=True,
    ):
        """
        Trains the network using mini-batch gradient descent:
        1. Shuffle the training data each epoch.
        2. Split it into small batches.
        3. For each batch: forward pass -> loss -> backward pass -> update weights.
        4. After each epoch, measure loss/accuracy on both train and
           validation data so we can watch learning progress.

        Returns a history dict with the per-epoch metrics (used later for
        the loss/accuracy plots).
        """
        num_samples = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        rng = np.random.default_rng(0)

        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                _, cache = self.forward(X_batch)
                grads = self.backward(y_batch, cache)
                self.update_params(grads, learning_rate)

            train_probs, _ = self.forward(X_train)
            val_probs, _ = self.forward(X_val)

            train_loss = cross_entropy_loss(train_probs, y_train)
            train_acc = accuracy(train_probs, y_train)
            val_loss = cross_entropy_loss(val_probs, y_val)
            val_acc = accuracy(val_probs, y_val)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch:2d}/{epochs} - "
                    f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                    f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                )

        return history
