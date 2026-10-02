"""
Matplotlib helpers for visualizing training progress and predictions.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(history, save_path="outputs/loss_accuracy.png"):
    """Plots training/validation loss and accuracy curves side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_title("Loss over training")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set_title("Accuracy over training")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sample_predictions(X, y_true, y_pred, save_path="outputs/sample_predictions.png", num_samples=10):
    """
    Shows a grid of digit images with their true label and the network's
    predicted label. Correct predictions are titled in green, mistakes in red.

    X: (N, 784) flattened images (values 0-1).
    """
    num_samples = min(num_samples, X.shape[0])
    cols = 5
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_samples):
        image = X[i].reshape(28, 28)
        correct = y_true[i] == y_pred[i]
        color = "green" if correct else "red"

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"true: {y_true[i]} / pred: {y_pred[i]}", color=color, fontsize=10)
        axes[i].axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
