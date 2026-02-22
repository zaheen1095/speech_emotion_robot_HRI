# #!/usr/bin/env python3
# """
# Make Final SER Figures (happy/sad) — from your REAL data

# Place your CSVs next to this script or pass paths via CLI flags.
# Outputs are written to ./figs_final/ by default.

# Required inputs (CSV schemas)

# 1) predictions.csv  (row = an utterance; pooled or just test split)
#    columns:
#      split          -> "val" or "test" (string; optional if you only have test)
#      y_true         -> 0 for happy, 1 for sad (int)
#      y_prob_sad     -> model probability for "sad" (float in [0,1])
#    example:
#      split,y_true,y_prob_sad
#      test,1,0.81
#      test,0,0.18
#    Used for: Confusion matrices (argmax vs thresholded) + Reliability diagram (val fold).

# 2) stress.csv  (aggregated metrics per SNR condition for the *same final model/op point*)
#    columns:
#      snr            -> either "clean" or numeric dB like 20,15,10,5 (string or number)
#      macro_f1       -> macro-F1 at that SNR using the *final operating point*
#    example:
#      snr,macro_f1
#      clean,0.879
#      20,0.865
#      15,0.852
#      10,0.834
#      5,0.780
#    Used for: Noise robustness curve.

# 3) per_corpus.csv  (aggregated metrics per corpus; optional, only if you want a CSV for the paper)
#    columns:
#      corpus,acc,macro_f1,recall_happy,recall_sad

# Command-line (optional):
#   python make_final_figures.py \
#     --preds predictions.csv \
#     --stress stress.csv \
#     --percorpus per_corpus.csv \
#     --threshold 0.57 \
#     --outdir figs_final

# Notes:
# - Matplotlib only, no seaborn. One plot per figure, default colors.
# - Reliability diagram bins=10 on the validation split if available; else on all data.
# - If your predictions.csv contains both "val" and "test", reliability is computed on val.
# """

# import argparse, os, math, csv
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# def load_preds(path):
#     df = pd.read_csv(path)
#     needed = {"y_true","y_prob_sad"}
#     if not needed.issubset(df.columns):
#         raise ValueError(f"predictions.csv missing columns {needed}")
#     if "split" not in df.columns:
#         df["split"] = "test"
#     return df

# def plot_confmat(y_true, y_pred, title, outpath):
#     cm = confusion_matrix(y_true, y_pred, labels=[0,1])
#     fig, ax = plt.subplots(figsize=(4.0,4.0), dpi=200)
#     im = ax.imshow(cm, interpolation='nearest')
#     ax.set_title(title)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_xticks([0,1]); ax.set_xticklabels(['happy','sad'])
#     ax.set_yticks([0,1]); ax.set_yticklabels(['happy','sad'])
#     for (i, j), v in np.ndenumerate(cm):
#         ax.text(j, i, str(v), ha='center', va='center')
#     plt.tight_layout()
#     fig.savefig(outpath, bbox_inches='tight')
#     plt.close(fig)

# def reliability_diagram(df, bins, outpath, title):
#     y = df["y_true"].to_numpy().astype(int)
#     p = df["y_prob_sad"].to_numpy().astype(float)
#     edges = np.linspace(0,1,bins+1)
#     ids = np.digitize(p, edges) - 1
#     bin_acc, bin_conf, bin_cnt = [], [], []
#     for b in range(bins):
#         m = ids == b
#         if m.any():
#             # turn probs into decisions at 0.5 solely for "accuracy within bin"
#             acc = (y[m] == (p[m] >= 0.5).astype(int)).mean()
#             conf = p[m].mean()
#             bin_acc.append(acc); bin_conf.append(conf); bin_cnt.append(int(m.sum()))
#         else:
#             bin_acc.append(np.nan); bin_conf.append(np.nan); bin_cnt.append(0)
#     total = len(y)
#     ece = sum((bin_cnt[b]/total) * abs((bin_acc[b]-bin_conf[b])) for b in range(bins) if bin_cnt[b]>0)
#     centers = 0.5*(edges[:-1] + edges[1:])

#     fig = plt.figure(figsize=(6.0,4.0), dpi=200)
#     plt.plot([0,1],[0,1])
#     plt.plot(centers, bin_acc, marker='o')
#     plt.title(f"{title}\nECE ≈ {ece:.3f}")
#     plt.xlabel("Confidence"); plt.ylabel("Accuracy")
#     plt.xlim(0,1); plt.ylim(0,1)
#     plt.grid(True, linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     fig.savefig(outpath, bbox_inches='tight')
#     plt.close(fig)

# def plot_snr_curve(stress_df, outpath):
#     # normalize x to numbers; treat "clean" as 25 dB for spacing on axis
#     def to_db(x):
#         try:
#             return float(x)
#         except:
#             return 25.0  # "clean"
#     stress_df = stress_df.copy()
#     stress_df["snr_db"] = stress_df["snr"].apply(to_db)
#     stress_df = stress_df.sort_values("snr_db", ascending=False)
#     fig = plt.figure(figsize=(6.0,4.0), dpi=200)
#     plt.plot(stress_df["snr_db"], stress_df["macro_f1"], marker='o')
#     plt.title("Noise Robustness (Macro-F1 vs SNR)")
#     plt.xlabel("SNR (dB)"); plt.ylabel("Macro-F1")
#     plt.grid(True, linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     fig.savefig(outpath, bbox_inches='tight')
#     plt.close(fig)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--preds", default="predictions.csv")
#     ap.add_argument("--stress", default="stress.csv")
#     ap.add_argument("--percorpus", default=None)
#     ap.add_argument("--threshold", type=float, default=0.57)
#     ap.add_argument("--bins", type=int, default=10)
#     ap.add_argument("--outdir", default="figs_final")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     # 1) Confusion matrices (test split) and 2) Reliability (val if available)
#     df = load_preds(args.preds)
#     # choose splits
#     df_test = df[df["split"].str.lower()=="test"] if "split" in df.columns else df
#     df_val  = df[df["split"].str.lower()=="val"]  if "split" in df.columns and (df["split"].str.lower()=="val").any() else None

#     y_true_test = df_test["y_true"].to_numpy().astype(int)
#     p_sad_test  = df_test["y_prob_sad"].to_numpy().astype(float)

#     y_pred_argmax  = (p_sad_test >= 0.5).astype(int)
#     y_pred_thresh  = (p_sad_test >= args.threshold).astype(int)

#     plot_confmat(y_true_test, y_pred_argmax,  "Confusion (clean) — Argmax",       os.path.join(args.outdir,"final_confmat_clean_argmax.png"))
#     plot_confmat(y_true_test, y_pred_thresh,  f"Confusion (clean) — Thresholded t*={args.threshold}", os.path.join(args.outdir,"final_confmat_clean_thresh.png"))

#     # Reliability: prefer validation split if provided
#     df_rel = df_val if df_val is not None and len(df_val)>0 else df_test
#     reliability_diagram(df_rel, args.bins, os.path.join(args.outdir,"final_reliability_curve.png"),
#                         "Reliability Diagram (Temp-scaled probs if applied)")

#     # 3) SNR curve
#     stress_df = pd.read_csv(args.stress)
#     if not {"snr","macro_f1"}.issubset(stress_df.columns):
#         raise ValueError("stress.csv must have columns: snr, macro_f1")
#     plot_snr_curve(stress_df, os.path.join(args.outdir,"final_snr_macroF1_curve.png"))

#     # 4) Per-corpus CSV is just passed through for your LaTeX table
#     if args.percorpus:
#         pc = pd.read_csv(args.percorpus)
#         pc.to_csv(os.path.join(args.outdir,"final_per_corpus_results.csv"), index=False)

#     print("Done. Wrote figures to:", args.outdir)

# if __name__ == "__main__":
#     main()
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_cm_like_train_model(cm, class_names=("happy", "sad"), title="Confusion Matrix", out="cm.png"):
    cm = np.array(cm, dtype=int)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    plot_cm_like_train_model([[585, 77],[202, 460]],
                             title="Confusion Matrix (Test Set) — Online Aug (Argmax)",
                             out="fig_8.7.png")

    plot_cm_like_train_model([[575, 87],[191, 471]],
                             title="Confusion Matrix (Test Set) — Online Aug (Thresholded)",
                             out="fig_8.8.png")
