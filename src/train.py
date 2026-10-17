"""
train.py: Trains classical machine learning models on MNIST with optional PCA and GridSearchCV.
"""

import argparse
import os
import time
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.data import load_mnist


def get_model(model_name):
    """Returns the requested model instance with default or base hyperparameters."""
    if model_name == "logistic":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "knn":
        return KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "svm":
        return SVC(C=5.0, gamma="scale", kernel="rbf", random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MNIST classification models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logistic", "knn", "rf", "svm", "all"],
        help="Model to train.",
    )
    parser.add_argument(
        "--pca", type=int, default=0, help="Number of PCA components (0 to disable)."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform GridSearchCV hyperparameter tuning on a subset.",
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X_train, X_test, y_train, y_test = load_mnist()

    models_to_train = (
        ["logistic", "knn", "rf", "svm"] if args.model == "all" else [args.model]
    )

    results_summary = {}

    for name in models_to_train:
        print(f"\n--- Training {name.upper()} ---")

        # Build pipeline steps
        steps = []
        if args.pca > 0:
            steps.append(("pca", PCA(n_components=args.pca, random_state=42)))

        base_model = get_model(name)

        # Optional tuning on a subset for heavy models to keep runtimes reasonable
        if args.tune:
            print(
                f"Tuning hyperparameters for {
                    name
                } using GridSearchCV on a 10k subset..."
            )
            subset_idx = np.random.choice(len(X_train), 10000, replace=False)
            X_sub, y_sub = X_train[subset_idx], y_train[subset_idx]

            # If PCA is active, transform subset for tuning
            if args.pca > 0:
                pca_temp = PCA(n_components=args.pca, random_state=42)
                X_sub_transformed = pca_temp.fit_transform(X_sub)
            else:
                X_sub_transformed = X_sub

            if name == "knn":
                param_grid = {"n_neighbors": [3, 5, 7]}
                grid = GridSearchCV(
                    KNeighborsClassifier(n_jobs=-1), param_grid, cv=3, n_jobs=-1
                )
            elif name == "svm":
                param_grid = {"C": [1.0, 5.0, 10.0],
                              "gamma": ["scale", "auto"]}
                grid = GridSearchCV(
                    SVC(kernel="rbf", random_state=42), param_grid, cv=3, n_jobs=-1
                )
            elif name == "logistic":
                param_grid = {"C": [0.1, 1.0, 10.0]}
                grid = GridSearchCV(
                    LogisticRegression(
                        max_iter=1000, random_state=42, n_jobs=-1),
                    param_grid,
                    cv=3,
                    n_jobs=-1,
                )
            elif name == "rf":
                param_grid = {"n_estimators": [
                    100, 200], "max_depth": [None, 20]}
                grid = GridSearchCV(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=3,
                    n_jobs=-1,
                )

            grid.fit(X_sub_transformed, y_sub)
            best_estimator = grid.best_estimator_
            print(f"Best parameters for {name}: {grid.best_params_}")

            if args.pca > 0:
                steps.append(("model", best_estimator))
                pipeline = Pipeline(steps)
            else:
                pipeline = best_estimator
        else:
            steps.append(("model", base_model))
            pipeline = Pipeline(steps)

        # Fit on full training data
        start_time = time.time()
        print(f"Fitting {name} on full training set ({
              len(X_train)} samples)...")
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")

        # Evaluate accuracy
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        print(
            f"Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {
                test_acc * 100:.2f}%"
        )

        # Save model pipeline
        model_filename = f"models/{name}_pca{args.pca}.joblib"
        joblib.dump(pipeline, model_filename)
        print(f"Saved model to {model_filename}")

        results_summary[name] = {
            "train_time": round(train_time, 2),
            "test_accuracy": round(test_acc * 100, 2),
        }

    # Save metrics summary json
    import json

    with open("results/metrics.json", "w") as f:
        json.dump(results_summary, f, indent=4)


if __name__ == "__main__":
    main()
