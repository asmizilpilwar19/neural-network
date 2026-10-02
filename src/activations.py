"""
Activation functions and their derivatives, implemented by hand with NumPy.

An activation function decides how "excited" a neuron gets from its input.
Without one, stacking layers would just be one big linear function - the
network couldn't learn curved decision boundaries, which digit recognition
needs.
"""

import numpy as np


def relu(x):
    """Rectified Linear Unit: keeps positive values, zeroes out negative ones.
    relu(x) = max(0, x)
    Used in the hidden layer because it is simple and trains fast."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Slope of ReLU: 1 where x > 0, 0 otherwise. Needed during backprop to
    know how much a hidden neuron's input contributed to the error."""
    return (x > 0).astype(x.dtype)


def softmax(x):
    """
    Turns a row of raw output scores ("logits") into a probability
    distribution over the 10 digit classes (all values 0-1, summing to 1).

    softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

    We subtract the row-wise max before exponentiating purely for numerical
    stability (so exp() never overflows) - it does not change the result,
    since softmax(x) == softmax(x - constant).
    """
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
