# aggregate_results.py
import os
import re
import pandas as pd
from pathlib import Path

RESULTS_ROOT = Path("results")

def parse_report(path):
    """Parse classification report text file."""
    with open(path, "r") as f:
        text = f.read()
    # Extract overall metrics with regex
    acc = re.search(r"Overall Accuracy:\s*([0-9.]+)", text)
    f1m = re.search(r"F1 Score \(Macro\):\s*([0-9.]+)", text)
    return {
        "accuracy": float(acc.group(1)) if acc else None,
        "f1_macro": float(f1m.group(1)) if f1m else None,
    }

def collect_results():
    rows = []
    for exp in RESULTS_ROOT.glob("*"):
        if not exp.is_dir():
            continue
        report_arg = exp / "classification_report_argmax.txt"
        report_thr = exp / "classification_report_thresholded.txt"
        if report_arg.exists():
            rows.append({
                "experiment": exp.name,
                "mode": "argmax",
                **parse_report(report_arg)
            })
        if report_thr.exists():
            rows.append({
                "experiment": exp.name,
                "mode": "thresholded",
                **parse_report(report_thr)
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = collect_results()
    out_csv = RESULTS_ROOT / "summary_all_experiments.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")
    print(df)
