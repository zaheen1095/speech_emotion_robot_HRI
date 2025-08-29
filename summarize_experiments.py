import pandas as pd
import glob, os, re

rows = []

def parse_report(txt):
    # try sklearn-style table lines
    acc = None
    f1m = None

    # 1) try the table "accuracy" row (e.g., "accuracy 0.9825 399")
    for line in txt.splitlines():
        if re.search(r'^\s*accuracy\s+', line, flags=re.I):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    acc = float(parts[-2])  # usually the value before support
                except:
                    pass
        if re.search(r'^\s*macro avg\s+', line, flags=re.I):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    f1m = float(parts[-2])  # macro F1 usually right before support
                except:
                    pass

    # 2) fallbacks for "Overall Accuracy: 0.9825" style lines
    if acc is None:
        m = re.search(r'Overall Accuracy:\s*([\d.]+)', txt, flags=re.I)
        if m:
            acc = float(m.group(1))
    if f1m is None:
        m = re.search(r'F1 Score \(Macro\):\s*([\d.]+)', txt, flags=re.I)
        if m:
            f1m = float(m.group(1))

    return acc, f1m

# ARGMAX reports in any subdir under results/
for path in glob.glob("results/**/classification_report_argmax.txt", recursive=True):
    exp = os.path.basename(os.path.dirname(path))
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        acc, f1m = parse_report(txt)
        rows.append({"experiment": exp, "mode": "argmax", "accuracy": acc, "f1_macro": f1m})
    except Exception as e:
        print(f"[skip] {path}: {e}")

# THRESHOLDED reports
for path in glob.glob("results/**/classification_report_thresholded.txt", recursive=True):
    exp = os.path.basename(os.path.dirname(path))
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        acc, f1m = parse_report(txt)
        rows.append({"experiment": exp, "mode": "thresholded", "accuracy": acc, "f1_macro": f1m})
    except Exception as e:
        print(f"[skip] {path}: {e}")

df = pd.DataFrame(rows)

# sort nicely: by experiment name, then mode (argmax first)
mode_order = {"argmax": 0, "thresholded": 1}
df["mode_ord"] = df["mode"].map(mode_order).fillna(9)
df = df.sort_values(["experiment", "mode_ord"]).drop(columns=["mode_ord"])

out_path = "results/summary_all_experiments.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"[done] wrote {out_path}")
print(df)
