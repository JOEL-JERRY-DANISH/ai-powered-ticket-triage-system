import pandas as pd

path = r'c:\Users\JOEL JERRY DANISH P\OneDrive\Desktop\hackathon\ticket-triage\data\ticket_dataset-tickets-multi-lang-5-2-50-version.csv'
df = pd.read_csv(path)

# ---- TARGET: type ----
print("=== TARGET: type ===")
n_type = df["type"].nunique()
print(f"  Unique values: {n_type}")
tc = df["type"].value_counts()
for k, v in tc.items():
    print(f"    {k:<30s}: {v}  ({v/len(df)*100:.1f}%)")

print()
print("=== TARGET: queue ===")
n_queue = df["queue"].nunique()
print(f"  Unique values: {n_queue}")
qc = df["queue"].value_counts()
for k, v in qc.items():
    print(f"    {k:<30s}: {v}  ({v/len(df)*100:.1f}%)")

print()
print("=== priority ===")
n_priority = df["priority"].nunique()
print(f"  Unique values: {n_priority}")
pc = df["priority"].value_counts()
for k, v in pc.items():
    print(f"    {k:<30s}: {v}  ({v/len(df)*100:.1f}%)")

print()
print("=== LANGUAGE ===")
n_lang = df["language"].nunique()
lc = df["language"].value_counts()
print(f"  Unique languages: {n_lang}")
for k, v in lc.items():
    print(f"    {k:<10s}: {v}  ({v/len(df)*100:.1f}%)")

print()
print("=== version column ===")
n_ver = df["version"].nunique()
print(f"  Unique values: {n_ver}")
vc = df["version"].value_counts()
for k, v in vc.items():
    print(f"    {k}: {v}")

print()
print("=== 5 EXAMPLE TEXT/LABEL PAIRS (body -> queue) ===")
pd.set_option("display.max_colwidth", 120)
sample = df[["subject", "body", "type", "queue"]].dropna(subset=["body"]).head(5)
for i, row in sample.iterrows():
    print(f"\n--- Row {i} ---")
    print(f"  subject : {str(row['subject'])[:80]}")
    print(f"  body    : {str(row['body'])[:120]}")
    print(f"  type    : {row['type']}")
    print(f"  queue   : {row['queue']}")

print()
print("=== LEAKAGE CHECK: answer vs queue ===")
# answer contains the reply — could it carry label info?
print("  'answer' column sample:")
for i, row in df[["answer", "queue"]].dropna().head(3).iterrows():
    print(f"    queue={row['queue']} | answer={str(row['answer'])[:100]}")

print()
print("=== TRAIN/TEST SPLIT CHECK: any split column? ===")
for col in df.columns:
    vals = df[col].unique()
    vals_lower = [str(v).lower() for v in vals]
    if any(x in ["train", "test", "val", "validation", "dev"] for x in vals_lower):
        print(f"  FOUND SPLIT COLUMN: {col} -> {vals}")
print("  (no output above means no split column found)")

print()
print("=== TAGS OVERVIEW ===")
for col in ["tag_1","tag_2","tag_3"]:
    print(f"  {col} unique values: {df[col].nunique()}")
    top = df[col].value_counts().head(5)
    for k, v in top.items():
        print(f"    {k}: {v}")
