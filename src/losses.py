"""
Loss and accuracy calculations.

The loss is a single number that says "how wrong were the predictions?".
Training works by nudging the network's weights to make this number
smaller over time.
"""

import numpy as np


def cross_entropy_loss(predictions, targets_one_hot):
    """
    Categorical cross-entropy: penalizes the network heavily when it is
    confident AND wrong, and lightly when it is unsure.

    For one example with true class c:
        loss = -log(predicted_probability_of_class_c)

    We average this over the whole batch. A small epsilon is added inside
    the log so we never compute log(0), which would be -infinity.
    """
    epsilon = 1e-12
    clipped = np.clip(predictions, epsilon, 1 - epsilon)
    num_samples = predictions.shape[0]
    per_sample_loss = -np.sum(targets_one_hot * np.log(clipped), axis=1)
    return np.mean(per_sample_loss)


def accuracy(predictions, targets_one_hot):
    """
    Fraction of examples where the network's top-scoring class matches the
    true class. E.g. 0.92 means 92% of digits were classified correctly.
    """
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets_one_hot, axis=1)
    return np.mean(predicted_classes == true_classes)
