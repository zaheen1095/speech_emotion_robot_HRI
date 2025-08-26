# testing_emotion.py — raw-audio eval using trained threshold (clean + extras)
import os, glob, argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report

from extract_features import extract_mfcc
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURE_SETTINGS, CLASSES, INFERENCE_SETTINGS, SAD_THRESHOLD_OVERRIDE, USE_ATTENTION

def _as_prob_threshold(thr):
    """Accept either probability (0..1) or 'logit-like' (>1.0) and return a probability."""
    if thr is None:
        return None
    t = float(thr)
    if t > 1.0:  # treat as logit
        return 1.0 / (1.0 + np.exp(-t))
    return max(0.0, min(1.0, t))

def _compute_input_dim():
    return FEATURE_SETTINGS['n_mfcc'] * (
        1 + int(FEATURE_SETTINGS.get('use_delta', False)) +
        int(FEATURE_SETTINGS.get('use_delta_delta', False))
    )

def _plot_cm(cm, classes, title, out_png):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=120)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True', xlabel='Predicted', title=title)
    # write counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", default="datasets/raw_audio/test",
                        help="Folder with class subdirs (e.g., happy/, sad/) containing .wav")
    parser.add_argument("--ckpt", default="models/best_model.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--results_dir", default="results",
                        help="Where to save CSV/plots")
    parser.add_argument("--quiet", action="store_true", help="Hide per-file prints")
    args = parser.parse_args()

    # Collect test pairs from folders (happy/ & sad/ with .wav files)
    pairs = []
    for cls_idx, cls_name in enumerate(CLASSES):
        wavs = glob.glob(os.path.join(args.test_root, cls_name, "**", "*.wav"), recursive=True)
        for wav in wavs:
            pairs.append((wav, cls_idx))
    if not pairs:
        raise SystemExit(f"No .wav files found under {args.test_root}/{{{','.join(CLASSES)}}}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dim = _compute_input_dim()
    sample_feats = extract_mfcc(audio_path=pairs[0][0])  # shape (T, D)
    input_dim = sample_feats.shape[-1]
    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES), use_attention=USE_ATTENTION).to(device)

    # Load checkpoint (support both dict-with-state and raw state_dict)
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[info] loaded checkpoint (epoch={ckpt.get('epoch','?')}, best_metric={ckpt.get('best_metric','?')}, "
              f"sad_thr={ckpt.get('sad_threshold','?')})")
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # Decide threshold (precedence: checkpoint → override → inference settings)
    ckpt_thr = None
    if isinstance(ckpt, dict) and "sad_threshold" in ckpt:
        try:
            ckpt_thr = float(ckpt["sad_threshold"])
        except Exception:
            ckpt_thr = None
    th_sad = _as_prob_threshold(ckpt_thr)
    if th_sad is None and SAD_THRESHOLD_OVERRIDE is not None:
        th_sad = _as_prob_threshold(SAD_THRESHOLD_OVERRIDE)
    if th_sad is None:
        th_sad = _as_prob_threshold(INFERENCE_SETTINGS.get("sad_threshold", 0.50))
    print(f"[info] using sad_threshold={th_sad:.2f} (probability space)")

    per_file = []
    y_true, y_pred_argmax, y_pred_thr = [], [], []
    sad_idx   = CLASSES.index('sad')
    happy_idx = CLASSES.index('happy')

    # Inference
    with torch.no_grad():
        for audio_path, true_idx in sorted(pairs):
            try:
                feats = extract_mfcc(audio_path=audio_path)  # (T, D)
            except Exception as e:
                print(f"[skip] {audio_path}: {e}")
                continue
            inp = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, D)
            logits = model(inp)                         # (1, C)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C,)

            pred_idx = int(np.argmax(probs))
            p_sad = float(probs[sad_idx])
            pred_thr = sad_idx if p_sad >= th_sad else happy_idx

            if not args.quiet:
                probs_str = " ".join([f"{CLASSES[i]}:{probs[i]:.2f}" for i in range(len(CLASSES))])
                print(f"{os.path.relpath(audio_path, args.test_root)} -> {probs_str} | "
                      f"argmax={CLASSES[pred_idx]} ({probs[pred_idx]:.2f}) | thr={CLASSES[pred_thr]}")

            per_file.append((audio_path, probs, pred_idx, pred_thr))
            y_true.append(true_idx)
            y_pred_argmax.append(pred_idx)
            y_pred_thr.append(pred_thr)

    # --- Metrics (argmax) ---
    os.makedirs(args.results_dir, exist_ok=True)
    y_true_arr = np.array(y_true, dtype=int)
    y_argmax   = np.array(y_pred_argmax, dtype=int)
    acc = (y_argmax == y_true_arr).mean()
    rec_h = recall_score(y_true_arr, y_argmax, pos_label=happy_idx, zero_division=0)
    rec_s = recall_score(y_true_arr, y_argmax, pos_label=sad_idx, zero_division=0)
    f1m   = f1_score(y_true_arr, y_argmax, average="macro", zero_division=0)
    cm    = confusion_matrix(y_true_arr, y_argmax, labels=[happy_idx, sad_idx])

    print("\n==================================================")
    print(" ARGMAX REPORT (Test)")
    print("==================================================")
    print(classification_report(y_true_arr, y_argmax, target_names=CLASSES, digits=4))
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1m:.4f}")

    # Save confusion matrix (argmax)
    _plot_cm(cm, CLASSES, "Confusion Matrix (Argmax)", os.path.join(args.results_dir, "cm_argmax.png"))

    # --- Metrics (thresholded) ---
    y_thr = np.array(y_pred_thr, dtype=int)
    acc_t = (y_thr == y_true_arr).mean()
    rec_h_t = recall_score(y_true_arr, y_thr, pos_label=happy_idx, zero_division=0)
    rec_s_t = recall_score(y_true_arr, y_thr, pos_label=sad_idx, zero_division=0)
    f1m_t   = f1_score(y_true_arr, y_thr, average="macro", zero_division=0)
    cm_t    = confusion_matrix(y_true_arr, y_thr, labels=[happy_idx, sad_idx])

    print("\n==================================================")
    print(" THRESHOLDED REPORT (Test)  [p_sad = sad_threshold]")
    print("==================================================")
    print(classification_report(y_true_arr, y_thr, target_names=CLASSES, digits=4))
    print(f"Overall Accuracy: {acc_t:.4f}")
    print(f"F1 Score (Macro): {f1m_t:.4f}")

    # Save confusion matrix (thresholded)
    _plot_cm(cm_t, CLASSES, "Confusion Matrix (Thresholded)", os.path.join(args.results_dir, "cm_thresholded.png"))

    # Save per-file CSV
    df = pd.DataFrame([
        {
            "file": os.path.relpath(p, args.test_root),
            "true": CLASSES[t],
            **{f"p_{CLASSES[i]}": float(prob[i]) for i in range(len(CLASSES))},
            "pred_argmax": CLASSES[a],
            "pred_thr":    CLASSES[b]
        }
        for (p, prob, a, b), t in zip(per_file, y_true)
    ])
    df.to_csv(os.path.join(args.results_dir, "per_file_results.csv"), index=False)

    # Save a text summary
    with open(os.path.join(args.results_dir, "testing_emotion_summary.txt"), "w") as f:
        f.write(f"Sad threshold used: {th_sad:.3f}\n")
        f.write("\n[ARGMAX]\n")
        f.write(classification_report(y_true_arr, y_argmax, target_names=CLASSES, digits=4))
        f.write(f"\nAccuracy: {acc:.4f}  Macro-F1: {f1m:.4f}\n")
        f.write("\n[THRESHOLDED]\n")
        f.write(classification_report(y_true_arr, y_thr, target_names=CLASSES, digits=4))
        f.write(f"\nAccuracy: {acc_t:.4f}  Macro-F1: {f1m_t:.4f}\n")

    print(f"\n[done] Wrote CSV/plots to: {args.results_dir}")

if __name__ == "__main__":
    main()
