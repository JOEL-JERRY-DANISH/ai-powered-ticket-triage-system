"""
evaluator.py
------------
Evaluates the multilingual ticket-triage queue model on the held-out test set.

Dataset : data/ticket_dataset-tickets-multi-lang-5-2-50-version.csv
Model   : models/ticket_queue_model.joblib
Target  : "queue" (10-class routing queue)

IMPORTANT: The train/test split is reproduced identically using:
    train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
This guarantees evaluation is on exactly the same held-out samples
that were withheld during training.

Features used (must match trainer.py):
  - body     -> TF-IDF
  - priority -> OneHotEncoder
  - language -> OneHotEncoder

Excluded: answer, type, subject, tag_1..tag_8, version

Run from project root:
    python evaluator/evaluator.py
"""

import json
import pathlib

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data" / "ticket_dataset-tickets-multi-lang-5-2-50-version.csv"
MODEL_PATH   = ROOT / "models" / "ticket_queue_model.joblib"
METRICS_PATH = ROOT / "metrics.json"

# ── Constants (must mirror trainer.py exactly) ─────────────────────────────
TARGET_COL   = "queue"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

TEXT_COL     = "body"
CAT_COLS     = ["priority", "language"]
ALL_FEATURES = [TEXT_COL] + CAT_COLS

# Baseline reference: random chance on 10 balanced classes = 10%
BASELINE_ACCURACY = 0.10
BASELINE_MACRO_F1 = 0.10

# ── Header ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Ticket-Triage -- Queue Model Evaluator")
print("=" * 60)

# ── 1. Load dataset (read-only, never modified) ────────────────────────────
print(f"\n[INFO] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
total_rows = len(df)
print(f"[INFO] Total rows loaded: {total_rows}")

# ── 2. Prepare X and y (same preprocessing as trainer.py) ─────────────────
X = df[ALL_FEATURES].copy()
y = df[TARGET_COL].astype(str)

for col in CAT_COLS:
    X[col] = X[col].fillna("unknown").astype(str)
X[TEXT_COL] = X[TEXT_COL].astype(str)

# ── 3. Reproduce IDENTICAL stratified split ────────────────────────────────
# Using exact same parameters as trainer.py -- guarantees same held-out set
print(f"\n[INFO] Reproducing stratified split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"[INFO] Training samples : {len(X_train)}")
print(f"[INFO] Test samples     : {len(X_test)}  (held-out, never seen during training)")

# ── 4. Verify class representation in test set ─────────────────────────────
print(f"\n[INFO] Test set class distribution ({TARGET_COL}):")
test_class_counts = y_test.value_counts().sort_values(ascending=False)
for label, count in test_class_counts.items():
    pct = count / len(y_test) * 100
    print(f"       {label:<35s}: {count:5d}  ({pct:.1f}%)")

# ── 5. Load trained model ──────────────────────────────────────────────────
print(f"\n[INFO] Loading model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run 'python trainer/trainer.py' first."
    )
pipeline = joblib.load(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# ── 6. Predict on test set only ────────────────────────────────────────────
print("\n[INFO] Running predictions on held-out test set ...")
y_pred = pipeline.predict(X_test)
print(f"[INFO] Predictions complete. ({len(y_pred)} samples)")

# ── 7. Compute metrics ─────────────────────────────────────────────────────
class_labels = sorted(list(set(y_test.tolist())))

accuracy    = accuracy_score(y_test, y_pred)
macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
macro_prec  = precision_score(y_test, y_pred, average="macro",    zero_division=0)
macro_rec   = recall_score(y_test, y_pred, average="macro",       zero_division=0)

precision_per = precision_score(y_test, y_pred, labels=class_labels, average=None, zero_division=0)
recall_per    = recall_score(y_test,    y_pred, labels=class_labels, average=None, zero_division=0)
f1_per        = f1_score(y_test,        y_pred, labels=class_labels, average=None, zero_division=0)
cm            = confusion_matrix(y_test, y_pred, labels=class_labels)

# ── 8. Build per-class dict ────────────────────────────────────────────────
per_class = {}
for i, label in enumerate(class_labels):
    per_class[label] = {
        "precision": round(float(precision_per[i]), 4),
        "recall":    round(float(recall_per[i]), 4),
        "f1":        round(float(f1_per[i]), 4),
    }

# ── 9. Print full results report ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION RESULTS")
print("=" * 60)
print(f"\n  Dataset            : {DATA_PATH.name}")
print(f"  Total dataset size : {total_rows}")
print(f"  Training samples   : {len(X_train)}")
print(f"  Test samples       : {len(X_test)}")
print(f"  Number of classes  : {len(class_labels)}")
print(f"  Model              : LogisticRegression (class_weight=balanced)")
print(f"  Features           : body (TF-IDF), priority (OHE), language (OHE)")
print()
print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Macro F1     : {macro_f1:.4f}")
print(f"  Weighted F1  : {weighted_f1:.4f}")
print(f"  Macro Prec.  : {macro_prec:.4f}")
print(f"  Macro Recall : {macro_rec:.4f}")

print(f"\n  Per-class metrics:")
print(f"  {'Class':<35s} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*67}")
for label, m in per_class.items():
    print(f"  {label:<35s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
print(f"  Labels: {class_labels}")
for i, row in enumerate(cm.tolist()):
    print(f"  {class_labels[i]:<35s}: {row}")

# ── 10. Comparison vs baseline ────────────────────────────────────────────
acc_delta = accuracy - BASELINE_ACCURACY
f1_delta  = macro_f1 - BASELINE_MACRO_F1

print("\n" + "=" * 60)
print("  COMPARISON vs RANDOM BASELINE (10-class)")
print("=" * 60)
print(f"  {'Metric':<15} {'Baseline':>10} {'New Model':>10} {'Delta':>10}")
print(f"  {'-'*47}")
print(f"  {'Accuracy':<15} {BASELINE_ACCURACY:>10.4f} {accuracy:>10.4f} {acc_delta:>+10.4f}")
print(f"  {'Macro F1':<15} {BASELINE_MACRO_F1:>10.4f} {macro_f1:>10.4f} {f1_delta:>+10.4f}")

if accuracy > BASELINE_ACCURACY + 0.10:
    verdict = "STRONG IMPROVEMENT over random baseline"
elif accuracy > BASELINE_ACCURACY + 0.02:
    verdict = "IMPROVEMENT over random baseline"
elif accuracy > BASELINE_ACCURACY:
    verdict = "Marginal improvement over random baseline"
else:
    verdict = "Still near or below random chance -- investigate data quality"

print(f"\n  Verdict: {verdict}")
print("=" * 60)

# ── 11. Save metrics.json ─────────────────────────────────────────────────
metrics = {
    "model_version":           "queue_multilingual_v1",
    "model_file":              str(MODEL_PATH),
    "metrics_file":            str(METRICS_PATH),
    "dataset":                 DATA_PATH.name,
    "target":                  TARGET_COL,
    "features_used":           ALL_FEATURES,
    "total_dataset_size":      int(total_rows),
    "training_sample_count":   int(len(X_train)),
    "test_sample_count":       int(len(X_test)),
    "split_method":            f"stratified train_test_split(test_size={TEST_SIZE}, random_state={RANDOM_STATE})",
    "num_classes":             int(len(class_labels)),
    "class_labels":            class_labels,
    "accuracy":                round(float(accuracy), 4),
    "macro_f1":                round(float(macro_f1), 4),
    "weighted_f1":             round(float(weighted_f1), 4),
    "macro_precision":         round(float(macro_prec), 4),
    "macro_recall":            round(float(macro_rec), 4),
    "baseline_accuracy":       BASELINE_ACCURACY,
    "baseline_macro_f1":       BASELINE_MACRO_F1,
    "accuracy_delta":          round(float(acc_delta), 4),
    "macro_f1_delta":          round(float(f1_delta), 4),
    "verdict":                 verdict,
    "per_class_metrics":       per_class,
    "confusion_matrix":        cm.tolist(),
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[INFO] metrics.json saved -> {METRICS_PATH}")
print("[INFO] Evaluation complete.")
print("=" * 60)
