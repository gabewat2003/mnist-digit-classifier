"""
data.py: Load, scale, and split the MNIST dataset.
"""

import os
import urllib.request
import warnings

import numpy as np

# Mirror of the original MNIST files, used if OpenML is unreachable.
# Same images and the same 60k/10k train/test split as OpenML's mnist_784.
NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE_DIR = os.path.join(os.path.expanduser("~"), "scikit_learn_data")


def _load_openml():
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # per-retry network warnings
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X.astype(np.float64), y.astype(int)


def _load_npz():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "mnist.npz")
    if not os.path.exists(path):
        print(f"Downloading {NPZ_URL}...")
        urllib.request.urlretrieve(NPZ_URL, path)
    with np.load(path, allow_pickle=False) as f:
        X = np.concatenate([f["x_train"], f["x_test"]]).reshape(-1, 784)
        y = np.concatenate([f["y_train"], f["y_test"]])
    return X.astype(np.float64), y.astype(int)


def load_mnist():
    """Load MNIST, scale pixels to [0, 1], and use the standard 60k/10k split.

    Tries OpenML first (cached under ~/scikit_learn_data after the first
    download) and falls back to the mnist.npz mirror if OpenML is down.
    """
    print("Loading MNIST...")
    try:
        X, y = _load_openml()
    except Exception as exc:  # network errors surface as several exception types
        print(f"OpenML unavailable ({type(exc).__name__}); using mnist.npz mirror.")
        X, y = _load_npz()

    X = X / 255.0
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
