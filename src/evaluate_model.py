"""
evaluate_model.py
-----------------
Loads the trained Random Forest model and generates:
  1. Confusion matrix heatmap
  2. ROC curve with AUC
  3. Feature importance bar chart (top 20 features)
  4. Prints full classification metrics

Outputs saved to outputs/
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_artifact(name, directory):
    path = os.path.join(directory, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}.pkl not found in {directory}")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake (0)", "Real (1)"],
        yticklabels=["Fake (0)", "Real (1)"],
        ax=ax, linewidths=0.5, linecolor="gray"
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Random Forest\nFake News Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Random Forest\nFake News Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved -> {save_path}")
    return roc_auc


def plot_feature_importance(model, vectorizer, save_path, top_n=20):
    importances = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563EB" if i % 2 == 0 else "#3B82F6" for i in range(top_n)]
    bars = ax.barh(range(top_n), top_importances[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Most Important Features — Random Forest", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, top_importances[::-1]):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart saved -> {save_path}")


def main():
    print("Loading model and vectorizer...")
    model = load_artifact("random_forest_model", MODELS_DIR)
    vectorizer = load_artifact("tfidf_vectorizer", MODELS_DIR)

    print("Loading test data...")
    X_test = load_artifact("X_test", DATA_PROC)
    y_test = load_artifact("y_test", DATA_PROC)

    # ── Predictions ────────────────────────────────────────────────────────
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("MODEL EVALUATION — TEST SET")
    print("=" * 55)
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (wtd)   : {prec:.4f}")
    print(f"  Recall (wtd)      : {rec:.4f}")
    print(f"  F1-Score (wtd)    : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  {'':10s}  Predicted Fake  Predicted Real")
    print(f"  {'Actual Fake':10s}  {cm[0,0]:^14d}  {cm[0,1]:^13d}")
    print(f"  {'Actual Real':10s}  {cm[1,0]:^14d}  {cm[1,1]:^13d}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # ── Plots ──────────────────────────────────────────────────────────────
    print("Generating evaluation plots...")
    plot_confusion_matrix(cm, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
    roc_auc = plot_roc_curve(y_test, y_prob, os.path.join(OUTPUTS_DIR, "roc_curve.png"))
    plot_feature_importance(model, vectorizer, os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    print(f"\n  ROC-AUC Score     : {roc_auc:.4f}")
    print("\nEvaluation complete! All plots saved to outputs/")


if __name__ == "__main__":
    main()
