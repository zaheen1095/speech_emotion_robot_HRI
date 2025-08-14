import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import FEATURE_SETTINGS, RESAMPLED_DIR, FEATURES_DIR
import noisereduce as nr
from utils import pad_sequence   # <-- added

def extract_mfcc(audio_path: str = None, array: np.ndarray = None, sr: int = None) -> np.ndarray:
    try:
        # --- Load ---
        if audio_path is not None:
            y, sr = librosa.load(audio_path, sr=FEATURE_SETTINGS['sample_rate'])
        else:
            assert array is not None and sr == FEATURE_SETTINGS['sample_rate'], \
                "When not using audio_path, you must pass array and matching sr"
            y = array

        # --- Safety: no NaNs/Infs ---
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Trim leading/trailing silence (keeps emotional core tighter) ---
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        # Fallback if trimming leaves too little audio
        if y_trim.size >= int(0.25 * sr):   # ≥ 250 ms remains
            y = y_trim
        # else keep original y

        # --- Optional safer denoise (leave commented unless necessary) ---
        # y = reduce_noise_safely(y, sr)

        # --- Pick highest-energy window instead of "first N seconds" ---
        max_duration = FEATURE_SETTINGS.get('max_duration', 4.0)
        target = int(max_duration * sr)
        if len(y) > target:
            hop = int(0.10 * sr)  # slide 100ms
            best_i, best_rms = 0, -1.0
            for i in range(0, len(y) - target, hop):
                rms = float(np.mean(np.abs(y[i:i+target])))
                if rms > best_rms:
                    best_rms, best_i = rms, i
            y = y[best_i:best_i + target]

        # --- Mel -> MFCC (+Δ/+ΔΔ) ---
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=FEATURE_SETTINGS['n_fft'],
            hop_length=FEATURE_SETTINGS['hop_length'],
            n_mels=FEATURE_SETTINGS['n_mels'],
            fmin=FEATURE_SETTINGS['fmin'],
            fmax=FEATURE_SETTINGS['fmax']
        )
        mel_db = librosa.power_to_db(mel_spec)

        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=FEATURE_SETTINGS['n_mfcc'])
        feats_list = [mfcc]
        if FEATURE_SETTINGS.get('use_delta', False):
            feats_list.append(librosa.feature.delta(mfcc))
        if FEATURE_SETTINGS.get('use_delta_delta', False):
            feats_list.append(librosa.feature.delta(mfcc, order=2))

        # (T, F)
        feats = np.vstack(feats_list).T

        # --- CMVN (per-utterance) ---
        mean = feats.mean(axis=0, keepdims=True)
        std  = feats.std(axis=0, keepdims=True) + 1e-8
        feats = (feats - mean) / std

        # --- Pad/trim once (consistent with config) ---
        feats = pad_sequence(feats, max_len=FEATURE_SETTINGS['max_len']).astype(np.float32)

        return feats

    except Exception as e:
        raise ValueError(f"Error processing {audio_path}: {str(e)}")

def normalize_length(features : np.ndarray, target_len=150) -> np.ndarray:
    """
    (Deprecated if embedded in extract_mfcc; kept for compatibility)
    """
    if features.shape[0] < target_len:
        pad_amt = target_len - features.shape[0]
        features = np.pad(features, ((0, pad_amt), (0, 0)))
    else:
        features = features[:target_len, :]
    return features

def process_audio_files(split: str):
    split_dir = RESAMPLED_DIR / split
    output_dir = FEATURES_DIR / split
    os.makedirs(output_dir, exist_ok=True)

    for emotion in tqdm(os.listdir(split_dir), desc=f"Processing {split}"):
        emotion_dir = split_dir / emotion
        if not emotion_dir.is_dir():     # <-- added guard
            continue
        os.makedirs(output_dir / emotion, exist_ok=True)

        for filename in os.listdir(emotion_dir):
            if filename.endswith(".wav"):
                input_path = emotion_dir / filename
                output_path = output_dir / emotion / f"{Path(filename).stem}.npy"

                try:
                    features = extract_mfcc(audio_path=str(input_path))
                    # Optional sanity warning (keeps your names & logic)
                    if features.shape[0] != FEATURE_SETTINGS['max_len']:
                        print(f" Warn: {filename} -> {features.shape[0]} frames (expected {FEATURE_SETTINGS['max_len']}).")
                    np.save(output_path, features)
                except ValueError as e:
                    print(f" Skipping {filename}: {str(e)}")

if __name__ == "__main__":
    process_audio_files("train")
    process_audio_files("test")
