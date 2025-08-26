# offline_augmentation.py
import os, random
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

from config import RAW_AUDIO_DIR, AUGMENTED_DIR

# --------- basic, safe augmentations ----------
def add_noise(y, snr_db=15):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    noise_rms = rms / (10**(snr_db/20))
    noise = np.random.normal(0, noise_rms, size=y.shape).astype(np.float32)
    out = (y + noise).astype(np.float32)
    # clamp a little to avoid weird writer overflows
    out = np.clip(out, -0.999, 0.999)
    return out, f"noiseSNR{snr_db}"

def gain(y, db=3):
    g = 10**(db/20)
    out = (y * g).astype(np.float32)
    out = np.clip(out, -0.999, 0.999)
    return out, f"gain{db:+d}"

def pitch(y, sr, steps=+2):
    out = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps).astype(np.float32)
    out = np.clip(out, -0.999, 0.999)
    return out, f"pitch{steps:+d}"

def stretch(y, rate=0.90):
    # Some librosa versions require 'rate' to be keyword-only
    try:
        out = librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:
        # Older signatures may allow positional; keep a fallback
        out = librosa.effects.time_stretch(y, rate)
    # pad/trim back to roughly original length
    if len(out) > len(y):
        out = out[:len(y)]
    else:
        out = np.pad(out, (0, len(y) - len(out)))
    out = np.clip(out.astype(np.float32), -0.999, 0.999)
    return out, f"stretch{rate:.2f}"

# choose a small set of strong, complementary ops
AUG_RECIPES = [
    lambda y, sr: add_noise(y, snr_db=10),
    lambda y, sr: add_noise(y, snr_db=15),
    lambda y, sr: pitch(y, sr, steps=+2),
    lambda y, sr: pitch(y, sr, steps=-2),
    lambda y, sr: stretch(y, rate=0.90),
    lambda y, sr: stretch(y, rate=1.10),
    lambda y, sr: gain(y, db=+3),
    lambda y, sr: gain(y, db=-3),
]

# how many variants per file (cap so you don’t blow up the dataset)
MAX_AUG_PER_FILE = 2  # pick any 2 distinct recipes per file

def _write_audio(path: Path, y, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)

def _copy_original(in_path: Path, out_root: Path, rel: Path):
    out_path = out_root / rel
    if not out_path.exists():
        y, sr = librosa.load(str(in_path), sr=None, mono=True)
        _write_audio(out_path, y, sr)
    return out_path

def _augment_file(in_path: Path, out_root: Path, rel: Path):
    y, sr = librosa.load(str(in_path), sr=None, mono=True)
    # choose distinct recipes for this file
    recipes = random.sample(AUG_RECIPES, k=min(MAX_AUG_PER_FILE, len(AUG_RECIPES)))
    base_stem = in_path.stem
    for fn in recipes:
        y_aug, tag = fn(y, sr)
        out_name = f"{base_stem}__aug-{tag}.wav"
        out_path = out_root / rel.parent / out_name
        if not out_path.exists():  # idempotent
            _write_audio(out_path, y_aug, sr)

def process_split(split="train"):
    in_root  = RAW_AUDIO_DIR / split
    out_root = AUGMENTED_DIR / split

    wav_exts = (".wav", ".WAV")
    files = [p for p in in_root.rglob("*") if p.suffix in wav_exts]

    if not files:
        print(f"[warn] No WAVs under {in_root}")
        return

    print(f"[info] Copying originals and creating augmentations from {in_root} → {out_root}")
    for in_path in files:
        # keep relative class/subdir structure
        rel = in_path.relative_to(in_root)
        # 1) copy original
        _copy_original(in_path, out_root, rel)
        # 2) write a few augmented variants (train only by default)
        if split == "train":
            _augment_file(in_path, out_root, rel)

if __name__ == "__main__":
    # Train: originals + a few augs; Test: originals only (no augs)
    process_split("train")
    process_split("test")
    print("[done] Augmented audio ready in datasets/augmented_audio/train/")
