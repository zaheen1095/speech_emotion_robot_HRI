# evaluate_testset.py  (neutral, no sad-bias)
import os, glob
import torch
import numpy as np
import pandas as pd

from extract_features import extract_mfcc
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURE_SETTINGS, CLASSES  ,INFERENCE_SETTINGS

TEST_ROOT = "datasets/raw_audio/test"   # adjust if needed
CKPT_PATH = "models/best_model.pt"

# --- collect wavs from folder names (ground truth = folder) ---
pairs = []  # (path, true_idx)
for cls_idx, cls_name in enumerate(CLASSES):
    for wav in glob.glob(os.path.join(TEST_ROOT, cls_name, "*.wav")):
        pairs.append((wav, cls_idx))

if not pairs:
    raise SystemExit(f"No wav files under {TEST_ROOT}/<{','.join(CLASSES)}>")

# --- build model ---
input_dim = FEATURE_SETTINGS['n_mfcc'] * (
    1 + int(FEATURE_SETTINGS.get('use_delta', False)) +
    int(FEATURE_SETTINGS.get('use_delta_delta', False))
)
model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES))

# load checkpoint (supports both raw state_dict and dict with 'model_state_dict')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1) Allowlist the legacy numpy scalar used inside the checkpoint
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
# 2) Try safe load again; if it still fails, fall back to classic
try:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
except Exception as e:
    print(f"[warn] Safe load failed ({e}); falling back to classic loader for this trusted file.")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    # allow non-strict in case shapes changed slightly
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print("load_state_dict() -> missing:", missing, "unexpected:", unexpected)
model.eval()

# --- evaluate (pure argmax, no thresholds, no forcing to sad) ---
conf_argmax = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
conf_thr    = np.zeros_like(conf_argmax)
th_sad = INFERENCE_SETTINGS.get("sad_threshold", 0.45)
per_file = []

with torch.no_grad():
    for audio_path, true_idx in sorted(pairs):
        try:
            x = extract_mfcc(audio_path=audio_path)
        except Exception as e:
            print(f"[skip] {audio_path}: {e}")
            continue

        inp = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, T, D]
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_argmax = int(np.argmax(probs))
        p_sad = float(probs[1])
        pred_thr = 1 if p_sad >= th_sad else 0 

        conf_argmax[true_idx, pred_argmax] += 1
        conf_thr[true_idx, pred_thr]       += 1
        per_file.append((audio_path, probs, pred_argmax, pred_thr))

# --- per-file outputs ---
for path, probs, pred_idx, pred_thr in per_file:
    probs_str = " ".join([f"{CLASSES[i]}:{probs[i]:.2f}" for i in range(len(CLASSES))])
    print(f"{os.path.basename(path)} -> {probs_str} | "
          f"argmax={CLASSES[pred_idx]} ({probs[pred_idx]:.2f}) "
          f"| thr={CLASSES[pred_thr]}")


def _print_metrics(name, cm):
    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / max(total, 1)

    print(f"\n=== {name} ===")
    print(f"Overall accuracy: {acc:.3f}  (correct {correct}/{total})")

    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"Recall[{cls}]: {rec:.3f}  ({tp}/{tp+fn})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "true\\pred -> " + " ".join([f"{c:>8}" for c in CLASSES])
    print(header)
    for i, cls in enumerate(CLASSES):
        row = " ".join([f"{cm[i, j]:>8d}" for j in range(len(CLASSES))])
        print(f"{cls:10}: {row}")


# call for both confusion matrices:
_print_metrics("A) ARGMAX", conf_argmax)
_print_metrics("B) THRESHOLD (p_sad ≥ sad_threshold)", conf_thr)


# --- save csv ---
os.makedirs("results", exist_ok=True)
df = pd.DataFrame([
    {
        "file": os.path.basename(p),
        **{f"p_{CLASSES[i]}": float(probs[i]) for i in range(len(CLASSES))},
        "pred_argmax": CLASSES[pred_idx],
        "pred_thr":   CLASSES[pred_thr]
    }
    for p, probs, pred_idx, pred_thr in per_file
])

df.to_csv("results/per_file_results.csv", index=False)
print("Wrote results/per_file_results.csv")


