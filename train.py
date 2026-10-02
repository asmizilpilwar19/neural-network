"""
Entry point: loads MNIST, trains the from-scratch neural network, prints
progress, and saves plots of the results.

Run with:
    python train.py
"""

import numpy as np

from src.data_loader import load_mnist
from src.neural_network import NeuralNetwork
from src.visualize import plot_sample_predictions, plot_training_history

# Hyperparameters - deliberately modest so this runs in well under a
# minute on a normal laptop CPU, with no GPU required.
HIDDEN_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.1

# Using a subset of MNIST's 60,000 training images keeps each epoch fast
# while still comfortably reaching ~95%+ accuracy. Set to None to use all.
TRAIN_SUBSET = 10000


def main():
    print("Loading MNIST (will download once and cache locally)...")
    X_train, y_train, X_test, y_test, test_labels = load_mnist(data_dir="data")

    if TRAIN_SUBSET is not None:
        X_train, y_train = X_train[:TRAIN_SUBSET], y_train[:TRAIN_SUBSET]

    print(f"Training set: {X_train.shape[0]} images, Test set: {X_test.shape[0]} images")

    network = NeuralNetwork(input_size=784, hidden_size=HIDDEN_SIZE, output_size=10)

    history = network.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    final_predictions = network.predict(X_test)
    final_accuracy = np.mean(final_predictions == test_labels)
    print(f"\nFinal test accuracy: {final_accuracy:.4f}")

    plot_training_history(history, save_path="outputs/loss_accuracy.png")
    plot_sample_predictions(
        X_test, test_labels, final_predictions, save_path="outputs/sample_predictions.png", num_samples=10
    )


if __name__ == "__main__":
    main()
