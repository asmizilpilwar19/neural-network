"""
Downloads the MNIST handwritten-digit dataset and turns it into plain
NumPy arrays. No machine-learning library is used anywhere in this file -
only `urllib` (to fetch the files), `gzip`/`struct` (to unpack MNIST's
binary format) and `numpy` (to hold the data).

MNIST file format (IDX), in short:
- Images file: a short header (magic number, item count, rows, cols)
  followed by one byte per pixel (0-255) for every image, row by row.
- Labels file: a short header followed by one byte per label (0-9).
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# Public mirror of the original MNIST files (the original host is unreliable).
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def _read_idx_images(path):
    """Parse an IDX3 image file into a (num_images, rows, cols) uint8 array."""
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        buffer = f.read(num_images * rows * cols)
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images


def _read_idx_labels(path):
    """Parse an IDX1 label file into a (num_labels,) uint8 array."""
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels


def load_mnist_raw(data_dir="data"):
    """
    Downloads MNIST (if not already cached in `data_dir`) and returns the
    raw arrays:
        train_images: (60000, 28, 28) uint8, pixel values 0-255
        train_labels: (60000,)         uint8, digit 0-9
        test_images:  (10000, 28, 28)  uint8
        test_labels:  (10000,)         uint8
    """
    os.makedirs(data_dir, exist_ok=True)

    cache_path = os.path.join(data_dir, "mnist.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached["train_images"],
            cached["train_labels"],
            cached["test_images"],
            cached["test_labels"],
        )

    train_images_path = _download_if_missing(data_dir, FILES["train_images"])
    train_labels_path = _download_if_missing(data_dir, FILES["train_labels"])
    test_images_path = _download_if_missing(data_dir, FILES["test_images"])
    test_labels_path = _download_if_missing(data_dir, FILES["test_labels"])

    train_images = _read_idx_images(train_images_path)
    train_labels = _read_idx_labels(train_labels_path)
    test_images = _read_idx_images(test_images_path)
    test_labels = _read_idx_labels(test_labels_path)

    np.savez_compressed(
        cache_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    return train_images, train_labels, test_images, test_labels


def one_hot_encode(labels, num_classes=10):
    """Turns a (N,) array of integer labels into an (N, num_classes) array
    of one-hot rows, e.g. label 3 -> [0,0,0,1,0,0,0,0,0,0]."""
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def load_mnist(data_dir="data", num_classes=10):
    """
    Returns MNIST ready for the neural network:
        X_train: (60000, 784) float64, pixel values scaled to [0, 1]
        y_train: (60000, 10)  one-hot encoded labels
        X_test:  (10000, 784) float64
        y_test:  (10000, 10)
        raw test labels are also returned (handy for plotting/accuracy):
        y_test_labels: (10000,) the original digit 0-9
    """
    train_images, train_labels, test_images, test_labels = load_mnist_raw(data_dir)

    num_train = train_images.shape[0]
    num_test = test_images.shape[0]

    # Flatten each 28x28 image into a 784-length vector and scale 0-255 -> 0-1.
    X_train = train_images.reshape(num_train, -1).astype(np.float64) / 255.0
    X_test = test_images.reshape(num_test, -1).astype(np.float64) / 255.0

    y_train = one_hot_encode(train_labels, num_classes)
    y_test = one_hot_encode(test_labels, num_classes)

    return X_train, y_train, X_test, y_test, test_labels
