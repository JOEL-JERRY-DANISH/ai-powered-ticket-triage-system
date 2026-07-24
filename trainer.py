"""
trainer.py
----------
Ticket-triage trainer for the multilingual dataset.

Dataset : data/ticket_dataset-tickets-multi-lang-5-2-50-version.csv
Target  : "queue"  (10-class routing queue)

Input features:
  - body      -> TF-IDF (1-2 grams, max_features=10000, sublinear_tf)
  - priority  -> OneHotEncoder
  - language  -> OneHotEncoder

Split   : stratified train_test_split (test_size=0.2, random_state=42)
          -- stratify on "queue" because classes are imbalanced (20x ratio)

Excluded (leakage / uninformative):
  - answer, type, subject, tag_1..tag_8, version

Run from project root:
    python trainer/trainer.py
"""

import pathlib

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "ticket_dataset-tickets-multi-lang-5-2-50-version.csv"
MODELS_DIR  = ROOT / "models"
MODEL_PATH  = MODELS_DIR / "ticket_queue_model.joblib"

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_COL   = "queue"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

TEXT_COL     = "body"
CAT_COLS     = ["priority", "language"]

ALL_FEATURES = [TEXT_COL] + CAT_COLS

# Safety: never include target as a feature
assert TARGET_COL not in ALL_FEATURES, "TARGET LEAKAGE DETECTED"

# ── Header ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Ticket-Triage -- Multilingual Queue Model Trainer")
print("=" * 60)
print(f"\n  Target   : {TARGET_COL}")
print(f"  Text     : {TEXT_COL}")
print(f"  Categ.   : {CAT_COLS}")
print(f"  Excluded : answer, type, subject, tag_1..tag_8, version")
print()

# ── 1. Load dataset ────────────────────────────────────────────────────────
print(f"[INFO] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
total_rows = len(df)
print(f"[INFO] Total rows loaded: {total_rows}")

# ── 2. Validate required columns ───────────────────────────────────────────
required = ALL_FEATURES + [TARGET_COL]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Required columns not found in dataset: {missing_cols}")
print(f"[INFO] All required columns present: {required}")

# ── 3. Validate 'body' has no missing values ───────────────────────────────
body_nulls = df[TEXT_COL].isnull().sum()
print(f"[INFO] Missing values in '{TEXT_COL}': {body_nulls}")
assert body_nulls == 0, f"'{TEXT_COL}' has {body_nulls} missing values -- investigate before training."

# ── 4. Prepare features and target ────────────────────────────────────────
X = df[ALL_FEATURES].copy()
y = df[TARGET_COL].astype(str)

# Fill missing values in categorical columns
for col in CAT_COLS:
    n_missing = X[col].isnull().sum()
    if n_missing > 0:
        print(f"[WARN] '{col}' has {n_missing} missing values -- filling with 'unknown'")
        X[col] = X[col].fillna("unknown").astype(str)
    else:
        X[col] = X[col].astype(str)

X[TEXT_COL] = X[TEXT_COL].astype(str)

# ── 5. Class distribution ──────────────────────────────────────────────────
print(f"\n[INFO] Full dataset class distribution ({TARGET_COL}):")
class_counts = y.value_counts().sort_values(ascending=False)
for label, count in class_counts.items():
    pct = count / len(y) * 100
    print(f"       {label:<35s}: {count:5d}  ({pct:.1f}%)")

# ── 6. Stratified train/test split ────────────────────────────────────────
print(f"\n[INFO] Splitting data: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"[INFO] Training samples : {len(X_train)}  ({len(X_train)/total_rows*100:.1f}%)")
print(f"[INFO] Test samples     : {len(X_test)}   ({len(X_test)/total_rows*100:.1f}%)")
print(f"[INFO] Split is stratified -- class proportions preserved in both sets.")

# ── 7. Build ColumnTransformer ─────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            ),
            TEXT_COL,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CAT_COLS,
        ),
    ],
    remainder="drop",
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
        ),
    ),
])

# ── 8. Train -- only on training data ─────────────────────────────────────
print(f"\n[INFO] Fitting pipeline on {len(X_train)} training samples ...")
print("[INFO] (Preprocessing fitted ONLY on training data -- no leakage)")
pipeline.fit(X_train, y_train)
print("[INFO] Model training complete.")

# ── 9. Save model ──────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\n[INFO] Model saved -> {MODEL_PATH}")
print("=" * 60)
