"""
train_model.py
--------------
Trains a Random Forest classifier on TF-IDF features.
Uses GridSearchCV for hyperparameter tuning with 5-fold cross-validation.

Outputs:
  - models/random_forest_model.pkl
  - outputs/evaluation_report.txt (training summary)
"""

import os
import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_splits():
    splits = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = os.path.join(DATA_PROC, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name}.pkl not found. Run src/preprocess.py first."
            )
        with open(path, "rb") as f:
            splits[name] = pickle.load(f)
    return splits


def main():
    print("Loading preprocessed data splits...")
    s = load_splits()
    X_train, y_train = s["X_train"], s["y_train"]
    X_val,   y_val   = s["X_val"],   s["y_val"]
    X_test,  y_test  = s["X_test"],  s["y_test"]

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val  : {X_val.shape[0]} samples")
    print(f"  Test : {X_test.shape[0]} samples")

    # ── Hyperparameter Grid ────────────────────────────────────────────────
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 20, 40],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    }

    print("\nRunning GridSearchCV (5-fold CV) — this may take a few minutes...")
    print("Parameter grid:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\nGridSearchCV completed in {elapsed:.1f}s")
    print(f"Best parameters : {grid_search.best_params_}")
    print(f"Best CV F1      : {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # ── Cross-validation scores on training set ────────────────────────────
    print("\nRunning 5-fold cross-validation on training data...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Validation set evaluation ──────────────────────────────────────────
    print("\nEvaluating on validation set...")
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(classification_report(y_val, y_val_pred, target_names=["Fake", "Real"]))

    # ── Test set evaluation ────────────────────────────────────────────────
    print("Evaluating on test set...")
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Test Accuracy      : {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=["Fake", "Real"]))

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nModel saved -> {model_path}")

    # ── Save training report ───────────────────────────────────────────────
    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST MODEL TRAINING REPORT\n")
        f.write("Fake News Detection — Nepal's Digital News Ecosystem\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Hyperparameters:\n")
        for k, v in grid_search.best_params_.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBest CV F1 Score (GridSearchCV): {grid_search.best_score_:.4f}\n")
        f.write(f"\n5-Fold Cross-Validation Accuracy (Training Set):\n")
        for i, sc in enumerate(cv_scores, 1):
            f.write(f"  Fold {i}: {sc:.4f}\n")
        f.write(f"  Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"\nValidation Set Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Set Accuracy      : {test_acc:.4f}\n")
        f.write("\nTest Set Classification Report:\n")
        f.write(classification_report(y_test, y_test_pred, target_names=["Fake", "Real"]))
        f.write("\nTest Set Confusion Matrix:\n")
        cm = confusion_matrix(y_test, y_test_pred)
        f.write(f"  TN={cm[0,0]}  FP={cm[0,1]}\n")
        f.write(f"  FN={cm[1,0]}  TP={cm[1,1]}\n")
    print(f"Training report saved -> {report_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
