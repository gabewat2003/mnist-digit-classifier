"""
train.py: Train classical models on MNIST with optional PCA and GridSearchCV tuning.

Examples:
    python -m src.train --model all --pca 50
    python -m src.train --model svm --pca 50 --tune
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import load_mnist

SEED = 42
MODELS = ["logistic", "knn", "rf", "svm"]
METRICS_PATH = "results/metrics.json"

# Hyperparameter grids searched with --tune (keys address the "model" pipeline step).
PARAM_GRIDS = {
    "logistic": {"model__C": [0.1, 1.0, 10.0]},
    "knn": {"model__n_neighbors": [1, 3, 5, 7]},
    "rf": {"model__n_estimators": [100, 200], "model__max_depth": [None, 20]},
    "svm": {"model__C": [1.0, 5.0, 10.0], "model__gamma": ["scale", 0.01, 0.05]},
}


def get_model(name):
    """Return the base estimator for a model name."""
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    if name == "svm":
        return SVC(C=5.0, gamma="scale", kernel="rbf", random_state=SEED)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name, n_components):
    steps = []
    if n_components > 0:
        steps.append(("pca", PCA(n_components=n_components, random_state=SEED)))
    steps.append(("model", get_model(name)))
    return Pipeline(steps)


def tune(pipeline, name, X_train, y_train, subset_size=10000):
    """Grid-search on a random subset, then return the pipeline with the best params."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), subset_size, replace=False)
    print(f"Tuning {name} with GridSearchCV (3-fold) on a {subset_size}-sample subset...")
    grid = GridSearchCV(pipeline, PARAM_GRIDS[name], cv=3, n_jobs=-1)
    grid.fit(X_train[idx], y_train[idx])
    print(f"Best params: {grid.best_params_} (CV accuracy {grid.best_score_:.4f})")
    pipeline.set_params(**grid.best_params_)
    return pipeline, grid.best_params_


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train MNIST classification models.")
    parser.add_argument("--model", required=True, choices=MODELS + ["all"])
    parser.add_argument("--pca", type=int, default=0,
                        help="Number of PCA components (0 disables PCA).")
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV on a 10k subset before the full fit.")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X_train, X_test, y_train, y_test = load_mnist()
    names = MODELS if args.model == "all" else [args.model]
    metrics = load_metrics()

    for name in names:
        print(f"\n--- Training {name} ---")
        pipeline = build_pipeline(name, args.pca)
        best_params = None
        if args.tune:
            pipeline, best_params = tune(pipeline, name, X_train, y_train)

        print(f"Fitting on full training set ({len(X_train)} samples)...")
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        print(f"Training took {train_time:.1f}s")

        start = time.perf_counter()
        test_acc = pipeline.score(X_test, y_test)
        predict_time = time.perf_counter() - start
        print(f"Test accuracy: {test_acc * 100:.2f}% (predicting 10k took {predict_time:.1f}s)")

        model_path = f"models/{name}_pca{args.pca}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved {model_path}")

        metrics[name] = {
            "pca_components": args.pca,
            "params": {k.removeprefix("model__"): v for k, v in (best_params or {}).items()},
            "train_time_s": round(train_time, 2),
            "predict_time_s": round(predict_time, 2),
            "test_accuracy": round(test_acc * 100, 2),
        }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nWrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
