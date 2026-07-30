"""
train_model.py
==============
Silent Voice — Model Training Script

Loads the hand-landmark dataset collected by collect_data.py, trains a
scikit-learn Random Forest classifier, evaluates performance, and saves
the trained model to model/classifier.pkl.

Usage
-----
    python train_model.py

Output
------
    - Console: accuracy, full classification report, and confusion matrix
    - File   : model/classifier.pkl  (joblib-serialised RandomForestClassifier)

Notes
-----
    * Random Forest is used without feature scaling because the algorithm is
      scale-invariant and MediaPipe already outputs normalised [0, 1] coords.
    * Stratified splitting ensures every class is represented in both splits,
      which matters for the smaller word-sign classes.
    * Reproducibility is ensured with random_state=42 throughout.

Author : Silent Voice Team
Date   : 2026
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH   = Path("data/landmarks.csv")
MODEL_DIR  = Path("model")
MODEL_PATH = MODEL_DIR / "classifier.pkl"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
N_ESTIMATORS  = 100     # number of decision trees in the forest
TEST_SIZE     = 0.20    # fraction of data held out for evaluation
RANDOM_STATE  = 42      # seed for reproducibility


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Read landmarks.csv and return feature matrix X and label vector y.

    The CSV is expected to have 63 numeric feature columns (x0,y0,z0,...,
    x20,y20,z20) and a final 'label' column produced by collect_data.py.

    Returns
    -------
    X : ndarray of shape (n_samples, 63)
    y : ndarray of shape (n_samples,) containing string class labels
    """
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} not found.  Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        print("[ERROR] 'label' column missing from landmarks.csv.")
        sys.exit(1)

    # Drop any rows where a landmark value is NaN (corrupt samples)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows with missing values.")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    return X, y


def print_class_distribution(y: np.ndarray) -> None:
    """Print how many samples exist per class, sorted alphabetically."""
    labels, counts = np.unique(y, return_counts=True)
    print(f"\n{'Class':<12} {'Samples':>8}")
    print("-" * 22)
    for lbl, cnt in zip(labels, counts):
        bar = "#" * (cnt // 100)      # mini-bar (1 # per 100 samples)
        print(f"  {lbl:<10} {cnt:>6}  {bar}")
    print(f"\n  Total  {len(y):>8}")


def print_confusion_matrix(cm: np.ndarray, classes: list[str]) -> None:
    """
    Pretty-print a confusion matrix to stdout.

    Rows = true label, columns = predicted label.  Only labels with any
    errors are shown to keep the output readable for 41 classes.
    """
    print("\n[RESULT] Confusion Matrix  (rows = true, cols = predicted)")
    print("         Only classes with at least one misclassification shown.\n")

    # Identify labels with errors
    error_mask  = np.any(cm != np.diag(np.diag(cm)), axis=1)
    show_labels = [lbl for lbl, show in zip(classes, error_mask) if show]

    if not show_labels:
        print("  -> Perfect classification on the test set! (No misclassifications)")
        return

    show_idx = [classes.index(l) for l in show_labels]
    sub_cm   = cm[np.ix_(show_idx, show_idx)]

    # Header row
    col_w = max(len(l) for l in show_labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in show_labels)
    print(header)

    for i, lbl in enumerate(show_labels):
        row = f"{lbl:<{col_w}}" + "".join(
            f"{sub_cm[i, j]:>{col_w}}" for j in range(len(show_labels))
        )
        print(row)


def main() -> None:
    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("[INFO] Loading dataset …")
    X, y = load_dataset()
    n_classes = len(np.unique(y))

    print(f"[INFO] Samples : {len(X)}")
    print(f"[INFO] Features: {X.shape[1]}  (21 landmarks × 3 axes)")
    print(f"[INFO] Classes : {n_classes}")
    print_class_distribution(y)

    # ── 2. Train / test split (stratified) ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,           # preserve class proportions in both splits
    )
    print(f"\n[INFO] Train set : {len(X_train)} samples")
    print(f"[INFO] Test set  : {len(X_test)}  samples")

    # ── 3. Train Random Forest ─────────────────────────────────────────────────
    print(f"\n[INFO] Training RandomForest "
          f"(n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}) …")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,            # use all available CPU cores
    )
    clf.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # ── 4. Evaluation ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"{'='*55}")

    # Full per-class precision / recall / F1
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix (compact form showing only misclassified classes)
    classes = sorted(clf.classes_)
    cm      = confusion_matrix(y_test, y_pred, labels=classes)
    print_confusion_matrix(cm, classes)

    # Feature importances — top 10 most discriminative landmark coords
    importances   = clf.feature_importances_
    feature_names = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
    top10_idx     = np.argsort(importances)[::-1][:10]
    print("\n[INFO] Top-10 most important features:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"  {rank:>2}. {feature_names[idx]:<6}  importance={importances[idx]:.4f}")

    # ── 5. Save model ──────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n[INFO] Model saved -> {MODEL_PATH.resolve()}")
    print("[INFO] Run predict_live.py or app.py to use the model.")


if __name__ == "__main__":
    main()
