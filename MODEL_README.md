# Ticket Type Classifier — Model README

**For Member 2 (API / Integration Developer)**

This document explains how to load the trained model and vectorizer, make predictions, and understand the output format.

---

## What Was Built

A **4-class ticket type classifier** that predicts the ITSM ticket category for an incoming support ticket.

| Property | Value |
|----------|-------|
| **Task** | Text classification (multiclass) |
| **Target column** | `type` |
| **Classes** | Change, Incident, Problem, Request |
| **Best model** | Linear SVM (LinearSVC) |
| **Accuracy** | **85.08%** |
| **Macro F1** | **0.8543** |
| **Dataset** | `ticket_dataset-tickets-multi-lang-5-2-50-version.csv` |
| **Languages** | English (57%) + German (43%) |

---

## Deliverables

All files are saved in the `models/` directory:

| File | Location | Description |
|------|----------|-------------|
| `model.pkl` | `models/model.pkl` | Trained LinearSVC classifier |
| `vectorizer.pkl` | `models/vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `type_metrics.json` | `type_metrics.json` | Full evaluation metrics |

---

## Prerequisites

```bash
pip install scikit-learn pandas numpy
```

> **Python version:** 3.9+  
> **scikit-learn version:** Must match the version used during training to safely unpickle. Check `type_metrics.json` or run `pip show scikit-learn`.

---

## How to Load the Model and Make a Prediction

### Basic Usage

```python
import pickle

# 1. Load the vectorizer and model
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Prepare the input text
#    Combine subject + body exactly as done during training:
subject = "Account Disruption"
body    = "I am writing to report a significant problem with the account management portal."
text    = (subject.strip() + " " + body.strip()).strip()

# 3. Vectorize the input (transform only, never fit)
X = vectorizer.transform([text])

# 4. Predict
predicted_type = model.predict(X)[0]
print("Predicted type:", predicted_type)
# Output: Predicted type: Incident
```

---

### If Subject is Missing

If a ticket has no subject, fill it with an empty string:

```python
subject = ""           # No subject available
body    = "My invoice is incorrect. I was charged twice."
text    = (subject.strip() + " " + body.strip()).strip()

X = vectorizer.transform([text])
predicted_type = model.predict(X)[0]
print("Predicted type:", predicted_type)
# Output: Predicted type: Incident
```

---

### Batch Prediction (Multiple Tickets)

```python
import pickle
import pandas as pd

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

tickets = [
    {"subject": "Billing Issue",       "body": "I was charged twice for my subscription."},
    {"subject": "App Crash",           "body": "The application keeps crashing on startup."},
    {"subject": "Password Reset",      "body": "Please reset my account password."},
    {"subject": "",                    "body": "Ich habe ein Problem mit meinem Konto."},
]

df = pd.DataFrame(tickets)
df["text"] = (df["subject"].fillna("").str.strip() + " " +
              df["body"].fillna("").str.strip()).str.strip()

X = vectorizer.transform(df["text"])
predictions = model.predict(X)

for ticket, pred in zip(tickets, predictions):
    print(f"  Body: {ticket['body'][:50]:<50s}  ->  {pred}")
```

---

## Output Classes

| Class | Meaning | Test set count |
|-------|---------|---------------|
| `Incident` | Unplanned interruption or degradation | 2,293 (40.1%) |
| `Request` | Request for information or standard change | 1,638 (28.6%) |
| `Problem` | Root cause investigation of one or more incidents | 1,203 (21.0%) |
| `Change` | Planned change to a service or system | 584 (10.2%) |

---

## Full Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | **85.08%** |
| Macro F1 | **0.8543** |
| Weighted F1 | 0.8476 |
| Macro Precision | 0.8617 |
| Macro Recall | 0.8497 |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Change | 0.97+ | 0.97+ | **0.9656** | 584 |
| Incident | ~0.85 | ~0.81 | **0.8279** | 2,293 |
| Problem | ~0.60 | ~0.66 | **0.6324** | 1,203 |
| Request | ~0.99 | ~0.99 | **0.9912** | 1,638 |

> **Note on Problem class:** The Incident ↔ Problem boundary is inherently ambiguous — many tickets describe a recurring issue (Problem) using incident-style language. This is expected and not a model defect.

---

## Model Comparison (Full Results)

Three classifiers were evaluated. Linear SVM was selected as the best:

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression | 82.30% | 0.8378 | 0.8255 |
| **Linear SVM ← Selected** | **85.08%** | **0.8543** | **0.8476** |
| Multinomial Naive Bayes | 82.07% | 0.8132 | 0.8102 |

---

## How the Text Feature Was Built

```
text = subject.strip() + " " + body.strip()
```

- If `subject` is `NaN` or empty → treated as `""`
- `body` had zero missing values in the training data
- **Always combine subject + body in this exact order** when calling the model

---

## Important Notes for Integration

1. **Always call `vectorizer.transform()` — never `fit_transform()`.**  
   `fit_transform()` would rebuild a new vocabulary and break predictions.

2. **Column order matters.** The vectorizer was fit on `subject + " " + body`. Reversing the order or omitting subject will degrade accuracy.

3. **Language support.** The model was trained on both English and German text. No special language preprocessing is needed — pass the raw text directly.

4. **The model does NOT return probabilities.** `LinearSVC` does not natively support `predict_proba()`. If confidence scores are required, use `model.decision_function(X)` to get raw margins, or switch to Logistic Regression at a small accuracy cost (~3%).

5. **Pickle version compatibility.** Load the model in the same Python + scikit-learn environment used during training. If the environment changes, retrain with `python train_type_classifier.py`.

---

## Quick Confidence Scores via Decision Function

If you need a relative confidence estimate (not a true probability):

```python
import numpy as np

decision_scores = model.decision_function(X)   # shape: (n_samples, n_classes)
classes = model.classes_

# Softmax to convert margins to pseudo-probabilities
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

pseudo_probs = softmax(decision_scores)

for i, cls in enumerate(classes):
    print(f"  {cls}: {pseudo_probs[0][i]:.4f}")
```

---

## Retraining

If you need to retrain (e.g., with a new dataset):

```bash
python train_type_classifier.py
```

This will overwrite `models/model.pkl`, `models/vectorizer.pkl`, and `type_metrics.json`.

---

## File Summary

```
ticket-triage/
├── models/
│   ├── model.pkl              # Trained LinearSVC model
│   └── vectorizer.pkl         # Fitted TF-IDF vectorizer (vocab size: 15,000)
├── type_metrics.json          # Full evaluation results (JSON)
└── train_type_classifier.py   # Training script (Member 1's deliverable)
```
