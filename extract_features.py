import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import FEATURE_SETTINGS, RESAMPLED_DIR, FEATURES_DIR, USE_EXTRA_FEATURES, EXTRA_FEATURES, PROSODY_SETTINGS, AUGMENTED_DIR
from utils import pad_sequence   # keep as-is
from ssl_frontend import SSLFrontend
# --- helpers ----------------------------------------------------

def _is_augmented_name(name: str) -> bool:
    n = name.lower()
    return (
        ".aug" in n or "_aug" in n or "-aug" in n or "__aug" in n or
        "noise" in n or "reverb" in n or "rir" in n or "pitch" in n or
        "tempo" in n or "speed" in n or "stretch" in n
    )

def _compute_prosody_TF(y, sr, T_target):
    H = PROSODY_SETTINGS["hop_length"]
    rms = librosa.feature.rms(y=y, frame_length=PROSODY_SETTINGS["frame_length"], hop_length=H)[0]
    S = np.abs(librosa.stft(y, n_fft=PROSODY_SETTINGS["frame_length"], hop_length=H)) ** 2
    spec_centroid  = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    spec_rolloff   = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=PROSODY_SETTINGS["rolloff_percent"])[0]
    f0 = librosa.yin(y,
                     fmin=PROSODY_SETTINGS["fmin_hz"],
                     fmax=PROSODY_SETTINGS["fmax_hz"],
                     sr=sr,
                     frame_length=PROSODY_SETTINGS["frame_length"],
                     hop_length=H)
    voiced_flag = ~np.isnan(f0)
    f0 = np.where(voiced_flag, f0, 0.0).astype(np.float32)
    voiced_flag = voiced_flag.astype(np.float32)

    bank = []
    for name in EXTRA_FEATURES:
        if name == "f0_hz": bank.append(f0)
        elif name == "voiced_flag": bank.append(voiced_flag)
        elif name == "rms": bank.append(rms)
        elif name == "spec_centroid": bank.append(spec_centroid)
        elif name == "spec_bandwidth": bank.append(spec_bandwidth)
        elif name == "spec_rolloff": bank.append(spec_rolloff)

    F = np.vstack(bank).astype(np.float32) if bank else np.zeros((0, f0.shape[0]), np.float32)

    T = F.shape[1]
    if T < T_target:
        pad = np.repeat(F[:, -1:], T_target - T, axis=1)
        F = np.concatenate([F, pad], axis=1)
    elif T > T_target:
        F = F[:, :T_target]
    return F.T  # (T_target, F_extra)

def extract_ssl_feats(wav, cfg):
    global _ssl
    if not hasattr(extract_ssl_feats, "_ssl"):
        extract_ssl_feats._ssl = SSLFrontend(cfg.SSL_MODEL, cfg.SSL_FREEZE, device="cuda" if torch.cuda.is_available() else "cpu")
    return extract_ssl_feats._ssl(wav)

def load_filelist(split="train"):
    """Return a list of audio file paths for the given split."""
    wav_exts = (".wav", ".WAV")
    in_root = RESAMPLED_DIR / split
    files = [p for p in in_root.rglob("*") if p.suffix in wav_exts]
    return files
# --- core extraction --------------------------------------------

def extract_mfcc(audio_path: str = None, array: np.ndarray = None, sr: int = None) -> np.ndarray:
    try:
        # load
        if audio_path is not None:
            y, sr = librosa.load(audio_path, sr=FEATURE_SETTINGS['sample_rate'])
        else:
            assert array is not None and sr == FEATURE_SETTINGS['sample_rate']
            y = array

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # trim
        y_trim, _ = librosa.effects.trim(y, top_db=FEATURE_SETTINGS["trim_db"])
        if y_trim.size >= int(0.25 * sr):
            y = y_trim

        # center-crop to max_duration
        max_duration = FEATURE_SETTINGS.get('max_duration', 5.0)
        target = int(max_duration * sr)
        if len(y) > target:
            start = (len(y) - target) // 2
            y = y[start:start + target]

        # mel -> mfcc (+delta, +delta2)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=FEATURE_SETTINGS['n_fft'],
            hop_length=FEATURE_SETTINGS['hop_length'],
            n_mels=FEATURE_SETTINGS['n_mels'],
            fmin=FEATURE_SETTINGS['fmin'],
            fmax=FEATURE_SETTINGS['fmax']
        )
        mel_db = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=FEATURE_SETTINGS['n_mfcc'])
        feats_list = [mfcc]
        if FEATURE_SETTINGS.get('use_delta', False):
            feats_list.append(librosa.feature.delta(mfcc))
        if FEATURE_SETTINGS.get('use_delta_delta', False):
            feats_list.append(librosa.feature.delta(mfcc, order=2))
        feats = np.vstack(feats_list).T  # (T, D)

        # CMVN per utt
        mean = feats.mean(axis=0, keepdims=True)
        std  = feats.std(axis=0, keepdims=True) + 1e-8
        feats = (feats - mean) / std

        # pad/trim frames
        feats = pad_sequence(feats, max_len=FEATURE_SETTINGS['max_len']).astype(np.float32)

        # B1 extras
        if USE_EXTRA_FEATURES:
            extra_TF = _compute_prosody_TF(y, sr, feats.shape[0])
            feats = np.concatenate([feats, extra_TF], axis=1)

        return feats

    except Exception as e:
        raise ValueError(f"Error processing {audio_path}: {str(e)}")

def process_audio_files(split: str):
    # >>> IMPORTANT: read **from AUGMENTED_DIR** so we include originals + *_aug variants
    src_root = AUGMENTED_DIR / split
    out_root = FEATURES_DIR  / split
    out_root.mkdir(parents=True, exist_ok=True)

    # sanity
    if not src_root.exists():
        print(f"[warn] {src_root} does not exist. Did you run offline_augmentation.py?")
        return

    wavs = [p for p in src_root.rglob("*.wav")]
    if not wavs:
        print(f"[warn] No WAVs under {src_root}")
        return

    n_total, n_aug = 0, 0
    for p in tqdm(wavs, desc=f"Extracting {split}"):
        rel = p.relative_to(src_root)            # keep class/subdir
        out_dir = (out_root / rel.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (p.stem + ".npy")

        try:
            feats = extract_mfcc(audio_path=str(p))
            np.save(out_path, feats)
            n_total += 1
            if _is_augmented_name(p.name):
                n_aug += 1
        except Exception as e:
            print(f" [skip] {p}: {e}")

    print(f"[done] {split}: saved {n_total} feature files to {out_root} (augmented: {n_aug})")

# --- optional legacy helpers kept for compatibility --------------

def normalize_length(features: np.ndarray, target_len=150) -> np.ndarray:
    if features.shape[0] < target_len:
        pad_amt = target_len - features.shape[0]
        features = np.pad(features, ((0, pad_amt), (0, 0)))
    else:
        features = features[:target_len, :]
    return features

if __name__ == "__main__":
    process_audio_files("train")
    process_audio_files("test")
