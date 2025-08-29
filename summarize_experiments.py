import pandas as pd
import glob, os, re

rows = []

def parse_report(txt):
    acc, f1m = None, None
    for line in txt.splitlines():
        if re.search(r'^\s*accuracy\s+', line, flags=re.I):
            parts = line.split()
            if len(parts) >= 2:
                try: acc = float(parts[-2])
                except: pass
        if re.search(r'^\s*macro avg\s+', line, flags=re.I):
            parts = line.split()
            if len(parts) >= 2:
                try: f1m = float(parts[-2])
                except: pass
    if acc is None:
        m = re.search(r'Overall Accuracy:\s*([\d.]+)', txt, flags=re.I)
        if m: acc = float(m.group(1))
    if f1m is None:
        m = re.search(r'F1 Score \(Macro\):\s*([\d.]+)', txt, flags=re.I)
        if m: f1m = float(m.group(1))
    return acc, f1m

# ARGMAX
for path in glob.glob("results/**/classification_report_argmax.txt", recursive=True):
    exp = os.path.basename(os.path.dirname(path))
    try:
        with open(path, encoding="utf-8") as f:
            acc, f1m = parse_report(f.read())
        rows.append({"experiment": exp, "mode": "argmax", "accuracy": acc, "f1_macro": f1m})
    except Exception as e:
        print(f"[skip] {path}: {e}")

# THRESHOLDED (two possible filenames)
for pattern in ["results/**/classification_report_thresholded.txt",
                "results/**/classification_report_threshold.txt"]:
    for path in glob.glob(pattern, recursive=True):
        exp = os.path.basename(os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                acc, f1m = parse_report(f.read())
            rows.append({"experiment": exp, "mode": "thresholded", "accuracy": acc, "f1_macro": f1m})
        except Exception as e:
            print(f"[skip] {path}: {e}")

df = pd.DataFrame(rows)

if df.empty:
    print("[warn] No reports found under results/**/. Did you run all tests?")
else:
    mode_order = {"argmax": 0, "thresholded": 1}
    df["mode_ord"] = df["mode"].map(mode_order).fillna(9)
    df = df.sort_values(["experiment", "mode_ord"]).drop(columns=["mode_ord"])

out_path = "results/summary_all_experiments.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"[done] wrote {out_path}")
print(df)
