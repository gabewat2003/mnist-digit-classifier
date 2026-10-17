"""
data.py: Loads, scales, splits MNIST dataset.
"""

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    """Fetch MNIST frm OpenML, scale pixels to [0, 1], and split train/test."""
    print("Loading MNIST dataset from OpenML")
    X, y = fetch_openml("mnist_784", version=1,
                        return_X_y=True, as_frame=False)
    y = y.astype(int)
    X = X / 255.0  # Scale pixel values to [0, 1]

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
