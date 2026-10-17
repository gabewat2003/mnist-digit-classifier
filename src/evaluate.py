"""
evaluate.py: Evaluates all saved models, generates unique plots for each, and prints a summary table.
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from src.data import load_mnist


def plot_pca_variance(X_train):
    """Plots cumulative explained variance ratio for PCA components."""
    print("Computing PCA explained variance...")
    pca = PCA().fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(cumsum, lw=2)
    plt.axhline(y=0.95, color="r", linestyle="--",
                label="95% Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True)
    plt.legend(loc="lower right")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/pca_variance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved results/pca_variance.png")


def evaluate_models():
    X_train, X_test, y_train, y_test = load_mnist()
    plot_pca_variance(X_train)

    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    if not model_files:
        print("No trained models found in models/ directory. Run train.py first.")
        return

    # Load training metrics if available
    metrics_path = "results/metrics.json"
    results_summary = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            results_summary = json.load(f)

    # Map internal names to README display names
    display_names = {
        "logistic": "Logistic regression",
        "knn": "k-NN",
        "rf": "Random forest",
        "svm": "SVM (RBF)",
    }

    print("\n" + "=" * 60)
    print(f"{'Model':<22} | {'Test accuracy':<15} | {'Train time':<12}")
    print("-" * 60)

    for file in sorted(model_files):
        model_name = file.split("_")[0]
        model_path = os.path.join("models", file)

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        test_acc = np.mean(y_pred == y_test) * 100

        train_time = results_summary.get(
            model_name, {}).get("train_time", "TODO")
        time_str = f"{train_time}s" if train_time != "TODO" else "TODO"
        acc_str = f"{test_acc:.2f}%"

        nice_name = display_names.get(model_name, model_name.capitalize())
        print(f"| {nice_name:<20} | {acc_str:<13} | {time_str:<10} |")

        # Generate unique confusion matrix for EACH model
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=np.arange(10))
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title(f"Confusion Matrix ({nice_name})")
        cm_path = f"results/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {cm_path}")

        # Generate unique misclassified grid for EACH model
        misclassified_idx = np.where(y_pred != y_test)[0]
        if len(misclassified_idx) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(8, 8))
            sample_indices = np.random.choice(
                misclassified_idx, size=min(9, len(misclassified_idx)), replace=False
            )
            for ax, idx in zip(axes.ravel(), sample_indices):
                img = X_test[idx].reshape(28, 28)
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                ax.set_title(f"True: {y_test[idx]} | Pred: {
                             y_pred[idx]}", fontsize=10)
            plt.suptitle(f"Misclassified Digits ({nice_name})", fontsize=14)
            plt.tight_layout()
            grid_path = f"results/{model_name}_misclassified_grid.png"
            plt.savefig(grid_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {grid_path}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_models()
