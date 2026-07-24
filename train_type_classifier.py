"""
train_type_classifier.py
------------------------
Complete ML pipeline for ticket TYPE classification.

Dataset   : data/ticket_dataset-tickets-multi-lang-5-2-50-version.csv
Target    : "type"  (4-class ITSM label: Incident, Request, Problem, Change)
Features  : subject + body  (concatenated into single text field)

Pipeline steps
--------------
1.  Load dataset
2.  Remove duplicate rows
3.  Handle missing values (subject, body, type)
4.  Combine subject + body into single text feature
5.  80/20 stratified split (random_state=42)
6.  TF-IDF Vectorizer
7.  Logistic Regression baseline
8.  Evaluate (accuracy, precision, recall, F1, confusion matrix, classification report)
9.  If accuracy < 85%: compare Linear SVM + Multinomial Naive Bayes
10. Select best model
11. Save: models/model.pkl  |  models/vectorizer.pkl  |  type_metrics.json

Run from project root:
    python train_type_classifier.py
"""

import json
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = pathlib.Path(__file__).resolve().parent
DATA_PATH    = ROOT / "data" / "ticket_dataset-tickets-multi-lang-5-2-50-version.csv"
MODELS_DIR   = ROOT / "models"
MODEL_PATH   = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH = ROOT / "type_metrics.json"

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_COL    = "type"
SUBJECT_COL   = "subject"
BODY_COL      = "body"
TEXT_COL      = "text"          # combined feature name
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
ACCURACY_THRESHOLD = 0.85       # If LR falls below this, compare other models

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def evaluate_model(name: str, model, X_test_tfidf, y_test, class_labels):
    """Return a dict of metrics for a given fitted model."""
    y_pred = model.predict(X_test_tfidf)

    acc        = accuracy_score(y_test, y_pred)
    macro_f1   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    macro_rec  = recall_score(y_test, y_pred, average="macro",       zero_division=0)
    cm         = confusion_matrix(y_test, y_pred, labels=class_labels)
    report     = classification_report(y_test, y_pred, labels=class_labels,
                                       zero_division=0, output_dict=True)

    print(f"\n  Model        : {name}")
    print(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  Macro Prec.  : {macro_prec:.4f}")
    print(f"  Macro Recall : {macro_rec:.4f}")

    return {
        "name":         name,
        "accuracy":     round(float(acc), 4),
        "macro_f1":     round(float(macro_f1), 4),
        "weighted_f1":  round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_labels": list(class_labels),
        "y_pred": list(y_pred),
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load dataset
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 1: Loading Dataset")
print(f"  Path : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
raw_rows = len(df)
print(f"  Raw rows  : {raw_rows}")
print(f"  Columns   : {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Remove duplicate rows
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 2: Removing Duplicate Rows")
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
n_dropped = n_before - n_after
print(f"  Rows before : {n_before}")
print(f"  Rows after  : {n_after}")
print(f"  Dropped     : {n_dropped} duplicate rows")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Handle missing values
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 3: Handling Missing Values")

for col in [SUBJECT_COL, BODY_COL, TARGET_COL]:
    n_miss = df[col].isnull().sum()
    print(f"  '{col}' missing: {n_miss}")

# subject: fill missing with empty string (body carries the signal)
df[SUBJECT_COL] = df[SUBJECT_COL].fillna("").astype(str)

# body: fill any rare missing with empty string
df[BODY_COL] = df[BODY_COL].fillna("").astype(str)

# type: drop rows where target is missing (critical)
n_before_target = len(df)
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
n_after_target = len(df)
print(f"  Rows dropped for missing '{TARGET_COL}': {n_before_target - n_after_target}")
print(f"  Final usable rows: {len(df)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Combine subject + body into single text feature
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 4: Combining subject + body")
# Concatenate with separator so TF-IDF sees both fields as continuous text.
# If subject is empty, it degrades cleanly to just body.
df[TEXT_COL] = (
    df[SUBJECT_COL].str.strip()
    + " "
    + df[BODY_COL].str.strip()
).str.strip()

total_usable = len(df)
print(f"  Combined text column '{TEXT_COL}' created.")
print(f"  Sample (row 0): {df[TEXT_COL].iloc[0][:100]} ...")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Class distribution + stratified split
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 5: Class Distribution + 80/20 Stratified Split")

X = df[TEXT_COL]
y = df[TARGET_COL]

print(f"\n  Full dataset class distribution ('{TARGET_COL}'):")
class_counts = y.value_counts().sort_values(ascending=False)
for label, count in class_counts.items():
    print(f"    {label:<12s}: {count:6d}  ({count/len(y)*100:.1f}%)")

class_labels = sorted(y.unique().tolist())
print(f"\n  Classes ({len(class_labels)}): {class_labels}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"\n  Training samples : {len(X_train)}  ({len(X_train)/total_usable*100:.1f}%)")
print(f"  Test samples     : {len(X_test)}   ({len(X_test)/total_usable*100:.1f}%)")
print(f"  random_state={RANDOM_STATE}, stratify=y  (proportions preserved)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — TF-IDF Vectorizer (fitted ONLY on training data)
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 6: TF-IDF Vectorization")
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    max_features=15000,
    sublinear_tf=True,
)

print("  Fitting TF-IDF on training data only (no leakage) ...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"  Vocabulary size   : {len(vectorizer.vocabulary_)}")
print(f"  Train matrix shape: {X_train_tfidf.shape}")
print(f"  Test  matrix shape: {X_test_tfidf.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Logistic Regression baseline
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 7: Training Logistic Regression (Baseline)")
lr_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
    solver="lbfgs",
    C=1.0,
    class_weight="balanced",
    multi_class="auto",
)
print("  Fitting LogisticRegression ...")
lr_model.fit(X_train_tfidf, y_train)
print("  Training complete.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — Evaluate Logistic Regression
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 8: Logistic Regression Evaluation")
lr_results = evaluate_model("Logistic Regression", lr_model,
                             X_test_tfidf, y_test, class_labels)

print(f"\n  Per-class metrics:")
print(f"  {'Class':<12s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"  {'-'*54}")
for label in class_labels:
    r = lr_results["classification_report"][label]
    print(f"  {label:<12s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")

print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
print(f"  Labels: {class_labels}")
for i, row in enumerate(lr_results["confusion_matrix"]):
    print(f"  {class_labels[i]:<12s}: {row}")

print(f"\n  Classification Report:")
print(classification_report(y_test,
                             lr_model.predict(X_test_tfidf),
                             labels=class_labels,
                             zero_division=0))

# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — Compare models if accuracy < 85%
# ══════════════════════════════════════════════════════════════════════════
all_results = [lr_results]
compared_models = False

if lr_results["accuracy"] < ACCURACY_THRESHOLD:
    banner(f"STEP 9: Accuracy {lr_results['accuracy']*100:.2f}% < {ACCURACY_THRESHOLD*100:.0f}% "
           f"— Comparing LinearSVC + Naive Bayes")
    compared_models = True

    # Linear SVM
    print("\n  Training LinearSVC ...")
    svm_model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, C=1.0)
    svm_model.fit(X_train_tfidf, y_train)
    svm_results = evaluate_model("Linear SVM (LinearSVC)", svm_model,
                                  X_test_tfidf, y_test, class_labels)
    all_results.append(svm_results)

    # Multinomial Naive Bayes
    # MNB requires non-negative features — TF-IDF with sublinear_tf is positive, OK
    print("\n  Training Multinomial Naive Bayes ...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_tfidf, y_train)
    nb_results = evaluate_model("Multinomial Naive Bayes", nb_model,
                                 X_test_tfidf, y_test, class_labels)
    all_results.append(nb_results)

else:
    banner(f"STEP 9: Accuracy {lr_results['accuracy']*100:.2f}% >= {ACCURACY_THRESHOLD*100:.0f}% "
           f"— Logistic Regression meets threshold, skipping comparison")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — Select best model
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 10: Selecting Best Model")

best_result = max(all_results, key=lambda r: r["macro_f1"])
best_name   = best_result["name"]
print(f"\n  Comparison summary:")
print(f"  {'Model':<30s} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
print(f"  {'-'*64}")
for r in all_results:
    marker = " <-- BEST" if r["name"] == best_name else ""
    print(f"  {r['name']:<30s} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}{marker}")

# Re-identify the best fitted model object
model_map = {"Logistic Regression": lr_model}
if compared_models:
    model_map["Linear SVM (LinearSVC)"] = svm_model
    model_map["Multinomial Naive Bayes"] = nb_model

best_model = model_map[best_name]
print(f"\n  SELECTED MODEL: {best_name}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11 — Save model.pkl, vectorizer.pkl, type_metrics.json
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 11: Saving Artifacts")

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"  model.pkl saved      -> {MODEL_PATH}")

# Save vectorizer
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"  vectorizer.pkl saved -> {VECTORIZER_PATH}")

# Build metrics dict
metrics = {
    "pipeline":          "type_classifier_v1",
    "dataset":           DATA_PATH.name,
    "target":            TARGET_COL,
    "features":          [SUBJECT_COL, BODY_COL],
    "combined_feature":  TEXT_COL,
    "total_rows_raw":    int(raw_rows),
    "total_rows_used":   int(total_usable),
    "duplicates_removed":int(n_dropped),
    "training_samples":  int(len(X_train)),
    "test_samples":      int(len(X_test)),
    "split_method":      f"stratified train_test_split(test_size={TEST_SIZE}, random_state={RANDOM_STATE})",
    "tfidf_max_features":15000,
    "tfidf_ngram_range": [1, 2],
    "num_classes":       int(len(class_labels)),
    "class_labels":      class_labels,
    "class_distribution":{
        label: int(count) for label, count in class_counts.items()
    },
    "best_model":        best_name,
    "model_file":        str(MODEL_PATH),
    "vectorizer_file":   str(VECTORIZER_PATH),
    "accuracy":          best_result["accuracy"],
    "macro_f1":          best_result["macro_f1"],
    "weighted_f1":       best_result["weighted_f1"],
    "macro_precision":   best_result["macro_precision"],
    "macro_recall":      best_result["macro_recall"],
    "baseline_accuracy_threshold": ACCURACY_THRESHOLD,
    "model_comparison":  compared_models,
    "all_model_results": [
        {
            "model":        r["name"],
            "accuracy":     r["accuracy"],
            "macro_f1":     r["macro_f1"],
            "weighted_f1":  r["weighted_f1"],
        }
        for r in all_results
    ],
    "per_class_metrics": {
        label: {
            "precision": round(best_result["classification_report"][label]["precision"], 4),
            "recall":    round(best_result["classification_report"][label]["recall"], 4),
            "f1":        round(best_result["classification_report"][label]["f1-score"], 4),
            "support":   int(best_result["classification_report"][label]["support"]),
        }
        for label in class_labels
    },
    "confusion_matrix": best_result["confusion_matrix"],
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"  type_metrics.json saved -> {METRICS_PATH}")

# ══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════
banner("FINAL REPORT")
print(f"""
  Dataset            : {DATA_PATH.name}
  Total rows (raw)   : {raw_rows}
  Duplicates removed : {n_dropped}
  Rows used          : {total_usable}
  Training samples   : {len(X_train)}
  Test samples       : {len(X_test)}
  Number of classes  : {len(class_labels)}
  Classes            : {class_labels}

  Best model         : {best_name}
  Accuracy           : {best_result['accuracy']:.4f}  ({best_result['accuracy']*100:.2f}%)
  Macro F1           : {best_result['macro_f1']:.4f}
  Weighted F1        : {best_result['weighted_f1']:.4f}
  Macro Precision    : {best_result['macro_precision']:.4f}
  Macro Recall       : {best_result['macro_recall']:.4f}

  Per-class F1:""")
for label in class_labels:
    f1_val = best_result["classification_report"][label]["f1-score"]
    print(f"    {label:<12s}: {f1_val:.4f}")

print(f"""
  Model file         : {MODEL_PATH}
  Vectorizer file    : {VECTORIZER_PATH}
  Metrics file       : {METRICS_PATH}
""")
print("=" * 62)
print("  Training pipeline complete.")
print("=" * 62)
