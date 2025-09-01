# cache_ssl_features.py (minimal diff)
import os, numpy as np, soundfile as sf, tqdm, torch
from pathlib import Path
from ssl_frontend import SSLFrontend
from config import FEATURE_SETTINGS, AUGMENTED_DIR   # <-- add
import argparse

def _ensure_mono16k(y, sr):
    ...
def _time_resize(TD, T_target):
    ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default=str(AUGMENTED_DIR),   # <-- NEW
                    help="Root with WAVs (e.g., datasets/augmented_audio)")
    ap.add_argument("--out_dir", type=str, default="datasets/features_ssl",
                    help="Where to write SSL features")
    ap.add_argument("--model",   type=str, default="wav2vec2-base")
    ap.add_argument("--freeze",  action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    ssl = SSLFrontend(model_name=args.model,
                      freeze=True,
                      device="cuda" if torch.cuda.is_available() else "cpu")

    T_target = int(FEATURE_SETTINGS["max_len"])

    for split in ["train", "test"]:
        in_split = Path(args.in_root) / split                 # <-- use in_root
        if not in_split.exists():
            print(f"[warn] {in_split} not found; skipping."); continue

        wavs = [p for p in in_split.rglob("*.wav")]
        if not wavs:
            print(f"[warn] No WAVs under {in_split}"); continue

        for wav in tqdm.tqdm(wavs, desc=f"SSL {split}"):
            rel = wav.relative_to(in_split)
            out_dir = out_root / split / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (wav.stem + ".npy")

            try:
                y, sr = sf.read(wav)
                # ensure mono/16k and shape -> (T_target, D_ssl)
                if sr != 16000 or (y.ndim > 1):
                    from librosa import resample
                    if y.ndim > 1: y = y.mean(axis=1)
                    if sr != 16000: y = resample(y, orig_sr=sr, target_sr=16000)
                feats = ssl(y.astype(np.float32))
                # linear time resize to match your model’s expected T
                T, D = feats.shape
                if T != T_target:
                    xp = np.linspace(0, T-1, num=T_target, dtype=np.float32)
                    lo = np.floor(xp).astype(int); hi = np.clip(lo+1, 0, T-1); w = xp - lo
                    feats = ((1-w)[:,None]*feats[lo] + w[:,None]*feats[hi]).astype(np.float32)
                np.save(out_path, feats)
            except Exception as e:
                print(f"[skip] {wav}: {e}")

        print(f"[done] Wrote SSL features to {out_root / split}")

if __name__ == "__main__":
    main()
