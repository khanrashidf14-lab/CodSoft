from __future__ import annotations
import argparse
import sys
from typing import Tuple, List

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

# ---------------- SAMPLE TRAINING DATA ----------------
SAMPLE_X = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Setosa
    [4.9, 3.0, 1.4, 0.2],   # Setosa
    [7.0, 3.2, 4.7, 1.4],   # Versicolor
    [6.4, 3.2, 4.5, 1.5],   # Versicolor
    [6.3, 3.3, 6.0, 2.5],   # Virginica
    [5.8, 2.7, 5.1, 1.9]    # Virginica
])

SAMPLE_y = np.array([
    "setosa",
    "setosa",
    "versicolor",
    "versicolor",
    "virginica",
    "virginica"
])


def build_pipeline(k: int | None = None) -> Tuple[Pipeline, dict]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])

    if k is None:
        param_grid = {
            "clf__n_neighbors": list(range(1, 11)),
            "clf__weights": ["uniform", "distance"]
        }
    else:
        param_grid = {
            "clf__n_neighbors": [k],
            "clf__weights": ["uniform", "distance"]
        }

    return pipe, param_grid


def train_and_evaluate(X: np.ndarray,
                       y: np.ndarray,
                       use_gridsearch: bool = True,
                       param_grid: dict | None = None,
                       cv: int = 5) -> Pipeline:
    """
    Train KNN pipeline on X,y. If use_gridsearch True, attempts GridSearchCV with cv,
    but adjusts or disables GridSearch when there are too few samples per class.
    Also automatically adjusts the n_neighbors search range so candidates are valid
    for the (approximate) training fold size to avoid NaN scores.
    """
    pipe, default_grid = build_pipeline()
    grid_params = param_grid.copy() if param_grid is not None else default_grid.copy()

    if not use_gridsearch:
        pipe.fit(X, y)
        return pipe

    # to determine minimum number of samples per class
    try:
        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min())
    except Exception:
        min_count = 0

    if min_count == 0:
        warnings.warn("Could not determine class counts reliably; proceeding with GridSearchCV as requested.")
    else:
        if cv > min_count:
            warnings.warn(
                f"Requested cv={cv} is greater than the smallest class size ({min_count}). "
                f"Reducing cv to {min_count} for GridSearchCV."
            )
            cv = min_count

    # to Estimate training set size per fold and restrict n_neighbors accordingly
    n_samples = X.shape[0]
    # approximate train size per fold = (cv-1)/cv * n_samples
    n_train_est = max(1, int((cv - 1) / cv * n_samples))
    max_k_allowed = max(1, min(10, n_train_est))  # cap at 10 by default, but don't exceed estimated train size

    if "clf__n_neighbors" in grid_params:
        original_range = grid_params["clf__n_neighbors"]
        try:
            grid_params["clf__n_neighbors"] = list(range(1, max_k_allowed + 1))
            print(f"Adjusted n_neighbors search range to 1..{max_k_allowed} (estimated train size per fold = {n_train_est}).")
        except Exception:
            grid_params["clf__n_neighbors"] = original_range

    if cv < 2:
        warnings.warn(
            "Not enough members per class to perform cross-validation. "
            "Skipping GridSearchCV and fitting the pipeline directly."
        )
        pipe.fit(X, y)
        return pipe

    #GridSearchCV with adjusted cv and adjusted n_neighbors grid
    grid = GridSearchCV(pipe, grid_params, cv=cv, n_jobs=-1, verbose=0)
    grid.fit(X, y)
    best = grid.best_estimator_
    print(f"\nBest parameters from GridSearchCV: {grid.best_params_}")
    #cross-validated score for the best estimator
    cv_scores = cross_val_score(best, X, y, cv=cv, n_jobs=-1)
    print(f"Cross-validated accuracy (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return best


def show_evaluation(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))


def parse_input_sample(s: str) -> np.ndarray | None:
    s = s.strip()
    if s.lower() in ("q", "quit", "exit"):
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated numbers (sepal_length,sepal_width,petal_length,petal_width).")
    try:
        floats = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError("All 4 values must be numeric.") from e
    return np.array([floats])

def interactive_predict_loop(model: Pipeline) -> None:
    clf: KNeighborsClassifier = model.named_steps["clf"]
    scaler: StandardScaler = model.named_steps["scaler"]
    classes = clf.classes_

    print("\nEnter samples as: sepal_length,sepal_width,petal_length,petal_width")
    print("Or type 'q' to quit.\n")
    try:
        while True:
            line = input("Sample> ").strip()
            if line.lower() in ("q", "quit", "exit"):
                print("Exiting interactive loop.")
                break
            try:
                X = parse_input_sample(line)
                if X is None:
                    break
            except ValueError as e:
                print("Invalid input:", e)
                continue

            X_scaled = scaler.transform(X)
            pred = clf.predict(X_scaled)[0]
            probs = clf.predict_proba(X_scaled)[0]

            prob_pairs = sorted(zip(classes, probs), key=lambda t: t[1], reverse=True)
            print(f"\nPredicted species: {pred}")
            print("Class probabilities:")
            for label, p in prob_pairs:
                print(f"  {label:12s}: {p:.3f}")

            k_used = clf.n_neighbors
            distances, indices = clf.kneighbors(X_scaled, n_neighbors=k_used)
            print(f"\n{k_used} nearest neighbors (distance, class):")
            neighbor_label = getattr(clf, "y", None) or getattr(clf, "_y", None)
            for d, idx in zip(distances[0], indices[0]):
                label_text = "unknown"
                if neighbor_label is not None:
                    try:
                        label_text = neighbor_label[idx]
                    except Exception:
                        label_text = "unknown"
                print(f"  idx={idx:3d}  dist={d:.3f}  label={label_text}")
            print("\n" + "-" * 40)
    except (KeyboardInterrupt, EOFError):
        print("\nInteractive session terminated.")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Advanced KNN Iris classifier (interactive).")
    parser.add_argument("--full", action="store_true", help="Use the full sklearn iris dataset (otherwise uses small sample).")
    parser.add_argument("--k", type=int, default=None, help="Fix k (n_neighbors) to this value instead of searching.")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds for GridSearch/CV.")
    parser.add_argument("--no-search", action="store_true", help="Don't run GridSearch; train with provided/ default params.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the trained model (joblib).")

    # When running in notebooks, allow kernel args by using parse_known_args()
    if argv is None:
        args, _unknown = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)

    if args.full:
        iris = load_iris()
        X = iris.data
        y = iris.target_names[iris.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        use_eval = True
    else:
        X_train, y_train = SAMPLE_X, SAMPLE_y
        X_test, y_test = None, None
        use_eval = False

    print("Training KNN model...")
    pipe, default_grid = build_pipeline(k=args.k)
    if args.no_search:
        final_model = train_and_evaluate(X_train, y_train, use_gridsearch=False)
    else:
        final_model = train_and_evaluate(X_train, y_train, use_gridsearch=True, param_grid=default_grid, cv=args.cv)

    if use_eval and X_test is not None:
        show_evaluation(final_model, X_test, y_test)

    interactive_predict_loop(final_model)

    if args.save:
        try:
            joblib.dump(final_model, args.save)
            print(f"Model saved to {args.save}")
        except Exception as e:
            print(f"Failed to save model: {e}")


if __name__ == "__main__":
    main()
