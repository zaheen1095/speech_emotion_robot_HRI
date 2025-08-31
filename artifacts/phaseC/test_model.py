# test_model.py
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_bilstm import CNNBiLSTM
from config import (
    FEATURES_DIR, CLASSES, MODEL_DIR, BATCH_SIZE, FEATURE_SETTINGS,
    USE_ATTENTION, DATASET_PREFIXES, DEFAULT_TEST_DATASETS
)
from config import CALIBRATION_DIR
import json
# -------------------------
# Dataset
# -------------------------
class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = np.load(self.feature_paths[idx])  # (T, D)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def infer_corpus_from_filename(path_str: str) -> str:
    base = os.path.basename(path_str).lower()
    key = base.split("_", 1)[0]
    return DATASET_PREFIXES.get(key, "UNKNOWN")

# -------------------------
# Helpers
# -------------------------
def load_test_data(selected_corpora):
    if selected_corpora is None:
        selected_corpora = DEFAULT_TEST_DATASETS
    X, y = [], []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'test' / emotion
        if not emotion_dir.exists():
            continue
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                full = str(emotion_dir / file)
                corpus = infer_corpus_from_filename(full)
                if (selected_corpora is None) or (corpus in selected_corpora):
                    X.append(full)
                    y.append(idx)
                # if corpus in selected_corpora:              # <-- filter here
                #     X.append(full)
                #     y.append(idx)

    if not X:
        raise SystemExit(
            f"No test features found for {selected_corpora} under {FEATURES_DIR}/test/<{','.join(CLASSES)}> (.npy)"
        )
    return X, y

def _save_confmat(cm, labels, out_path, title):
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------
# Main test
# -------------------------
def test(checkpoint_path=None, test_datasets=None):
    # Default to your requested set if not provided
    selected = test_datasets or DEFAULT_TEST_DATASETS
    print(f"\nEvaluating only on: {selected}")

    print("\n🔎 Loading test data...")
    X_test, y_test = load_test_data(selected)
    test_loader = DataLoader(
        FeatureDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    with torch.no_grad():
        sample_x, _ = next(iter(test_loader))
    input_dim = sample_x.shape[-1]
    print(f"Inferred input_dim from test features: {input_dim}")

    # ----- Model -----
    print("\n🧠 Loading trained model...")
    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES), use_attention=USE_ATTENTION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint_path or (Path(MODEL_DIR) / "best_model.pt"))
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        sad_threshold = float(checkpoint.get('sad_threshold', 0.50))
        print(f"Loaded checkpoint (epoch={checkpoint.get('epoch','?')}, "
              f"best_metric={checkpoint.get('best_metric','?')}, "
              f"sad_threshold={sad_threshold:.3f})")
    else:
        model.load_state_dict(checkpoint)
        sad_threshold = 0.50
        print(f"Loaded plain state_dict; using default sad_threshold={sad_threshold:.2f}")
    
    model.to(device)
    model.eval()

    T = 1.0
    # indices for class-order safety
    try:
        sad_idx = CLASSES.index('sad')
        tfile = Path(CALIBRATION_DIR) / "temperature.json"
        if tfile.exists():
            T = float(json.load(open(tfile, "r"))["temperature"])
            print(f"[C3.2] Using temperature scaling T={T:.3f}")
    except Exception as e:
        print(f"[C3.1] Threshold not overridden ({e}); using checkpoint/default.)")
    except ValueError:
        raise SystemExit("Class 'sad' not found in CLASSES; thresholded report requires a 'sad' class.")
    happy_idx = CLASSES.index('happy') if 'happy' in CLASSES else (1 - sad_idx)

    print("\n🚀 Running inference on test set...")
    y_true, y_pred_argmax = [], []
    all_logits = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)              # [B, C]
            if T != 1.0:
                logits = logits / T
            preds = logits.argmax(dim=1)        # argmax
            y_true.extend(labels.cpu().tolist())
            y_pred_argmax.extend(preds.cpu().tolist())
            all_logits.append(logits.cpu())

    y_true = np.array(y_true, dtype=int)
    logits_all = torch.cat(all_logits, dim=0)            # [N, C]
    probs_all = F.softmax(logits_all, dim=1).numpy()     # [N, C]

    # -------------------------
    # ARGMAX REPORT
    # -------------------------
    print("\n" + "="*50)
    print(" ARGMAX REPORT (Test)")
    print("="*50)
    acc = accuracy_score(y_true, y_pred_argmax)
    f1_macro = f1_score(y_true, y_pred_argmax, average='macro')
    f1_weighted = f1_score(y_true, y_pred_argmax, average='weighted')
    report = classification_report(y_true, y_pred_argmax, target_names=CLASSES, digits=4)
    print(report)
    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    results_dir = Path(args.results_dir or "results")
    results_dir.mkdir(exist_ok=True, parents=True)
    with open(results_dir / "classification_report_argmax.txt", "w") as f:
        f.write(report)
        f.write(f"\n\nOverall Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n")

    cm_arg = confusion_matrix(y_true, y_pred_argmax, labels=list(range(len(CLASSES))))
    _save_confmat(cm_arg, CLASSES, results_dir / "confusion_matrix_argmax.png",
                  "Confusion Matrix (Test – Argmax)")

    # -------------------------
    # THRESHOLDED REPORT (binary sad vs happy)
    # -------------------------
    print("\n" + "="*50)
    print(" THRESHOLDED REPORT (Test)  [p_sad = sad_threshold]")
    print("="*50)
    p_sad = probs_all[:, sad_idx]
    y_pred_thr = np.where(p_sad >= sad_threshold, sad_idx, happy_idx)

    acc_t = accuracy_score(y_true, y_pred_thr)
    f1_macro_t = f1_score(y_true, y_pred_thr, average='macro')
    f1_weighted_t = f1_score(y_true, y_pred_thr, average='weighted')
    report_t = classification_report(y_true, y_pred_thr, target_names=CLASSES, digits=4)
    print(report_t)
    print(f"\nOverall Accuracy: {acc_t:.4f}")
    print(f"F1 Score (Macro): {f1_macro_t:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted_t:.4f}")

    with open(results_dir / "classification_report_thresholded.txt", "w") as f:
        f.write(report_t)
        f.write(f"\n\nOverall Accuracy: {acc_t:.4f}\n")
        f.write(f"F1 Score (Macro): {f1_macro_t:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1_weighted_t:.4f}\n")
        f.write(f"\nUsed sad_threshold={sad_threshold:.3f}\n")

    cm_thr = confusion_matrix(y_true, y_pred_thr, labels=list(range(len(CLASSES))))
    _save_confmat(cm_thr, CLASSES, results_dir / "confusion_matrix_thresholded.png",
                  f"Confusion Matrix (Test – Thresholded @ {sad_threshold:.2f})")

    # -------------------------
    # B4: Per-corpus breakdown (Argmax)
    # -------------------------
    from collections import defaultdict
    print("\n==== Per-corpus breakdown (Argmax) ====")
    corpus_to_idx = defaultdict(list)
    for i, fp in enumerate(X_test):
        corpus_to_idx[infer_corpus_from_filename(fp)].append(i)

    for corpus, idxs in sorted(corpus_to_idx.items()):
        yt = y_true[idxs]
        yp = np.array(y_pred_argmax, dtype=int)[idxs]
        f1m = f1_score(yt, yp, average="macro")
        print(f"{corpus:8s} | N={len(idxs):4d} | Macro-F1={f1m:.3f}")

    # Save raw probabilities/file order
    np.save(results_dir / "probs_test.npy", probs_all)
    with open(results_dir / "file_order.txt", "w") as f:
        for p in X_test:
            f.write(p + "\n")

    print("\nResults saved to 'results/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt (defaults to MODEL_DIR/best_model.pt)")
    parser.add_argument("--test_datasets", type=str, default="",
                        help="Comma-separated list to evaluate (e.g. IEMOCAP,CREMA-D,JL,RAVDESS,TESS). "
                             "If omitted, uses DEFAULT_TEST_DATASETS from config.")
    parser.add_argument("--results_dir", type=str, default="results",
                    help="Where to save evaluation results")

    args = parser.parse_args()
    chosen = [s.strip() for s in args.test_datasets.split(",") if s.strip()] or None
    test(checkpoint_path=args.checkpoint, test_datasets=chosen)
