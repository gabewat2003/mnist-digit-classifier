"""
evaluate.py: Evaluate every saved model, save its confusion matrix and a grid of
misclassified digits, and print a summary table with the most-confused digit pairs.

Usage:
    python -m src.evaluate
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

from src.data import load_mnist  # noqa: E402

SEED = 42
DISPLAY_NAMES = {
    "logistic": "Logistic regression",
    "knn": "k-NN",
    "rf": "Random forest",
    "svm": "SVM (RBF)",
}


def plot_pca_variance(X_train):
    """Plot cumulative explained variance and report how much 50 components keep."""
    print("Computing PCA explained variance...")
    cumsum = np.cumsum(PCA(random_state=SEED).fit(X_train).explained_variance_ratio_)
    n95 = int(np.searchsorted(cumsum, 0.95) + 1)
    print(f"50 components keep {cumsum[49] * 100:.1f}% of variance; "
          f"95% needs {n95} components.")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumsum) + 1), cumsum, lw=2)
    plt.axhline(0.95, color="r", linestyle="--", label="95% explained variance")
    plt.axvline(50, color="gray", linestyle=":", label="50 components")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig("results/pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close()


def top_confusions(cm, k=3):
    """Return the k most frequent (true, predicted, count) off-diagonal entries."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    return [(int(t), int(p), int(off[t, p])) for t, p in zip(*np.unravel_index(flat, off.shape))]


def plot_confusion_matrix(cm, name, nice):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(
        ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=False)
    ax.set_title(f"Confusion matrix ({nice})")
    fig.savefig(f"results/{name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(X_test, y_test, y_pred, name, nice):
    wrong = np.flatnonzero(y_pred != y_test)
    if len(wrong) == 0:
        return
    rng = np.random.default_rng(SEED)
    sample = rng.choice(wrong, size=min(9, len(wrong)), replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), sample):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"True: {y_test[idx]} | Pred: {y_pred[idx]}", fontsize=10)
    fig.suptitle(f"Misclassified digits ({nice})", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"results/{name}_misclassified_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_models():
    os.makedirs("results", exist_ok=True)
    model_files = sorted(f for f in os.listdir("models") if f.endswith(".joblib")) \
        if os.path.isdir("models") else []
    if not model_files:
        print("No trained models in models/. Run `python -m src.train --model all --pca 50` first.")
        return

    X_train, X_test, y_train, y_test = load_mnist()
    plot_pca_variance(X_train)

    metrics_path = "results/metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    rows = []
    for file in model_files:
        name = file.split("_pca")[0]
        nice = DISPLAY_NAMES.get(name, name)
        model = joblib.load(os.path.join("models", file))
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test) * 100
        cm = confusion_matrix(y_test, y_pred)

        plot_confusion_matrix(cm, name, nice)
        plot_misclassified(X_test, y_test, y_pred, name, nice)

        confusions = top_confusions(cm)
        entry = metrics.setdefault(name, {})
        entry["test_accuracy"] = round(acc, 2)
        entry["top_confusions"] = [
            {"true": t, "pred": p, "count": c} for t, p, c in confusions]
        rows.append((nice, acc, entry.get("train_time_s"), confusions))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n| Model | Test accuracy | Train time | Most confused (true -> pred) |")
    print("|---|---|---|---|")
    for nice, acc, t, conf in rows:
        t_str = f"{t:.1f}s" if t is not None else "n/a"
        conf_str = ", ".join(f"{a}->{b} ({c})" for a, b, c in conf)
        print(f"| {nice} | {acc:.2f}% | {t_str} | {conf_str} |")


if __name__ == "__main__":
    evaluate_models()
