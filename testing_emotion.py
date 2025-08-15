# evaluate_testset.py
import os, glob
import torch
import numpy as np
from extract_features import extract_mfcc
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURE_SETTINGS, CLASSES
from config import FEATURE_SETTINGS, CLASSES, INFERENCE_SETTINGS

TEST_ROOT = "datasets/raw_audio/test"   # adjust if needed

# collect files from folder names (ground truth = folder)
pairs = []  # (path, true_idx)
for cls_idx, cls_name in enumerate(CLASSES):
    for wav in glob.glob(os.path.join(TEST_ROOT, cls_name, "*.wav")):
        pairs.append((wav, cls_idx))

if not pairs:
    raise SystemExit(f"No wav files under {TEST_ROOT}/<{','.join(CLASSES)}>")

# build model
input_dim = FEATURE_SETTINGS['n_mfcc'] * (
    1 + FEATURE_SETTINGS['use_delta'] + FEATURE_SETTINGS['use_delta_delta']
)
model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# eval
conf_mat = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
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
        pred_idx = int(np.argmax(probs))
        conf_mat[true_idx, pred_idx] += 1

        # optional: quick warning if filename prefix disagrees with folder label
        base = os.path.basename(audio_path).lower()
        if "ravdess_" in base and ("happy" in base or "sad" in base):
            prefix = "happy" if "happy" in base else "sad"
            if prefix != CLASSES[true_idx]:
                print(f"[warn] label mismatch? folder={CLASSES[true_idx]} fileprefix={prefix} -> {audio_path}")

        per_file.append((audio_path, probs, pred_idx))

# print per-file results
for path, probs, pred_idx in per_file:
    probs_str = " ".join([f"{CLASSES[i]}:{probs[i]:.2f}" for i in range(len(CLASSES))])
    print(f"{os.path.basename(path)} -> {probs_str} | pred={CLASSES[pred_idx]} ({probs[pred_idx]:.2f})")

# metrics
total = conf_mat.sum()
correct = np.trace(conf_mat)
acc = correct / max(total, 1)

print("\n=== METRICS ===")
print(f"Overall accuracy: {acc:.3f}  (correct {correct}/{total})")

for i, cls in enumerate(CLASSES):
    tp = conf_mat[i, i]
    fn = conf_mat[i, :].sum() - tp
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"Recall[{cls}]: {rec:.3f}  ({tp}/{tp+fn})")

print("\nConfusion matrix (rows=true, cols=pred):")
header = "true\\pred -> " + " ".join([f"{c:>8}" for c in CLASSES])
print(header)
for i, cls in enumerate(CLASSES):
    row = " ".join([f"{conf_mat[i,j]:>8d}" for j in range(len(CLASSES))])
    print(f"{cls:10}: {row}")
