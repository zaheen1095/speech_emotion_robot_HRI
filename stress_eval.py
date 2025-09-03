# stress_eval.py
import os, math, argparse, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, MODEL_DIR, BATCH_SIZE, USE_ATTENTION
# reuse what your test_model already has
from test_model import FeatureDataset, load_test_data  # <- these exist in your file
from config import DEFAULT_TEST_DATASETS
import pandas as pd
# ---------- tiny helpers ----------
def load_checkpoint_model(input_dim, device, ckpt_path=None):
    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES), use_attention=USE_ATTENTION)
    ckpt_path = Path(ckpt_path or (MODEL_DIR / "best_model.pt"))
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def infer_input_dim(sample_npy_path):
    x = np.load(sample_npy_path)  # (T, D)
    return x.shape[-1]

# --- simple feature-domain perturbations (proxy for C2.1) ---
def add_noise_for_snr(x, snr_db):
    # x: (T, D) feature matrix; add white noise so SNR ~ snr_db
    p_signal = (x**2).mean() + 1e-12
    target = p_signal / (10.0**(snr_db/10.0))
    noise = np.random.randn(*x.shape).astype(np.float32)
    p_noise = (noise**2).mean() + 1e-12
    scale = math.sqrt(target / p_noise)
    return x + scale * noise

def time_stretch_feat(x, rate):
    # stretch/compress along time axis using linear interp (feature proxy)
    T, D = x.shape
    new_T = max(1, int(round(T / rate)))
    src_idx = np.linspace(0, T-1, num=new_T).astype(np.float32)
    lo = np.floor(src_idx).astype(int)
    hi = np.clip(lo+1, 0, T-1)
    w = src_idx - lo
    out = (1-w)[:, None]*x[lo] + w[:, None]*x[hi]
    return out.astype(np.float32)

class FeatureDatasetWithPerturb(Dataset):
    def __init__(self, base_paths, labels, mode="clean", value=None):
        self.paths = base_paths
        self.labels = labels
        self.mode = mode
        self.value = value

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        x = np.load(self.paths[i]).astype(np.float32)    # (T, D)
        if self.mode == "snr" and self.value is not None:
            x = add_noise_for_snr(x, self.value)
        elif self.mode == "stretch" and self.value is not None:
            x = time_stretch_feat(x, self.value)
        y = self.labels[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def evaluate(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    return acc, f1m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(MODEL_DIR / "best_model.pt"))
    ap.add_argument("--results_csv", type=str, default="results/stress_summary.csv")

    ap.add_argument("--dataset", type=str, required=True,
                    help="Dataset name (e.g., CREMA-D, IEMOCAP, TESS, SAVEE, JL, CREMA-D).")

    ap.add_argument("--snr_db", type=int, default=None,
                    help="Additive white noise SNR in dB (feature-space proxy).")
    ap.add_argument("--tempo", type=float, default=None,
                    help="Tempo/stretch factor, e.g., 0.9 or 1.1 (feature-space proxy).")
    ap.add_argument("--pitch", type=int, default=None,
                    help="Pitch shift in semitones (NOT supported in feature-space; see note).")

    args = ap.parse_args()

    # --- select only the requested dataset
    selected = [args.dataset]
    X_test, y_test = load_test_data(selected_corpora=selected)

    # infer input dim and load model
    sample_dim = infer_input_dim(X_test[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(sample_dim, device, args.checkpoint)

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    rows = []

    # always evaluate clean first
    clean_ds = FeatureDataset(X_test, y_test)
    clean_ld = DataLoader(clean_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    acc, f1m = evaluate(model, clean_ld, device)
    rows.append({"condition": "clean", "param": "", "accuracy": acc, "f1_macro": f1m})

    ran_any = False

    # if a single SNR is requested, run just that
    if args.snr_db is not None:
        ds = FeatureDatasetWithPerturb(X_test, y_test, mode="snr", value=args.snr_db)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        acc, f1m = evaluate(model, ld, device)
        rows.append({"condition": "snr_db", "param": args.snr_db, "accuracy": acc, "f1_macro": f1m})
        ran_any = True

    # if a single tempo is requested, run just that
    if args.tempo is not None:
        ds = FeatureDatasetWithPerturb(X_test, y_test, mode="stretch", value=args.tempo)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        acc, f1m = evaluate(model, ld, device)
        rows.append({"condition": "stretch_rate", "param": args.tempo, "accuracy": acc, "f1_macro": f1m})
        ran_any = True

    # pitch not supported in feature space (needs waveform re-extraction)
    if args.pitch is not None:
        print("[note] --pitch requested, but pitch shift isn’t supported in this feature-space script. "
              "Skip or use a waveform-based stress script.")
        ran_any = True  # mark that user asked for something, even if we can’t run it here

    # if no specific perturbation flags were passed, run the full grid
    if not ran_any:
        for snr in [20, 15, 10, 5]:
            ds = FeatureDatasetWithPerturb(X_test, y_test, mode="snr", value=snr)
            ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            acc, f1m = evaluate(model, ld, device)
            rows.append({"condition": "snr_db", "param": snr, "accuracy": acc, "f1_macro": f1m})

        for rate in [0.9, 1.1]:
            ds = FeatureDatasetWithPerturb(X_test, y_test, mode="stretch", value=rate)
            ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            acc, f1m = evaluate(model, ld, device)
            rows.append({"condition": "stretch_rate", "param": rate, "accuracy": acc, "f1_macro": f1m})

    # save summary
   
    df = pd.DataFrame(rows)
    df.to_csv(args.results_csv, index=False)
    print(f"[done] wrote {args.results_csv}")
    print(df)

if __name__ == "__main__":
    main()
