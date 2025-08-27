import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import FEATURE_SETTINGS, RESAMPLED_DIR, FEATURES_DIR,USE_EXTRA_FEATURES, EXTRA_FEATURES, PROSODY_SETTINGS
import noisereduce as nr
from utils import pad_sequence   # keep as-is


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
        y_trim, _ = librosa.effects.trim(y, top_db=FEATURE_SETTINGS["trim_db"])
        if y_trim.size >= int(0.25 * sr):   # ≥ 250 ms remains
            y = y_trim

        # --- Optional safer denoise (leave commented unless necessary) ---
        # y = reduce_noise_safely(y, sr)

        # --- Pick highest-energy window (target = FEATURE_SETTINGS['max_duration']) ---
        max_duration = FEATURE_SETTINGS.get('max_duration', 5.0)
        target = int(max_duration * sr)
        # if len(y) > target:
        #     hop = int(0.10 * sr)  # slide 100ms
        #     best_i, best_rms = 0, -1.0
        #     for i in range(0, len(y) - target, hop):
        #         rms = float(np.mean(np.abs(y[i:i+target])))
        #         if rms > best_rms:
        #             best_rms, best_i = rms, i
        #     y = y[best_i:best_i + target]
        if len(y) > target:
            start = (len(y) - target) // 2
            y = y[start:start + target]

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

        if USE_EXTRA_FEATURES:
            extra_TF = _compute_prosody_TF(y, sr, feats.shape[0])  # (T, F_extra)
            feats = np.concatenate([feats, extra_TF], axis=1)         # early fusion

        return feats

    except Exception as e:
        raise ValueError(f"Error processing {audio_path}: {str(e)}")

def normalize_length(features: np.ndarray, target_len=150) -> np.ndarray:
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
        if not emotion_dir.is_dir():     # guard
            continue
        os.makedirs(output_dir / emotion, exist_ok=True)

        for filename in os.listdir(emotion_dir):
            if filename.endswith(".wav"):
                input_path = emotion_dir / filename
                output_path = output_dir / emotion / f"{Path(filename).stem}.npy"

                try:
                    features = extract_mfcc(audio_path=str(input_path))
                    if features.shape[0] != FEATURE_SETTINGS['max_len']:
                        print(f" Warn: {filename} -> {features.shape[0]} frames (expected {FEATURE_SETTINGS['max_len']}).")
                    np.save(output_path, features)
                except ValueError as e:
                    print(f" Skipping {filename}: {str(e)}")

def _compute_prosody_TF(y, sr, T_target):
    H = PROSODY_SETTINGS["hop_length"]
    # RMS
    rms = librosa.feature.rms(y=y, frame_length=PROSODY_SETTINGS["frame_length"], hop_length=H)[0]  # [T']
    # Spectral stats
    S = np.abs(librosa.stft(y, n_fft=PROSODY_SETTINGS["frame_length"], hop_length=H)) ** 2
    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=PROSODY_SETTINGS["rolloff_percent"])[0]
    # (Optional) spectral flux
    # spec_flux = np.sqrt(((np.diff(S, axis=1).clip(min=0))**2).sum(axis=0)) ; spec_flux = np.r_[0, spec_flux]

    # Pitch (YIN): returns F0 (Hz) with NaN for unvoiced
    f0 = librosa.yin(y,
                     fmin=PROSODY_SETTINGS["fmin_hz"],
                     fmax=PROSODY_SETTINGS["fmax_hz"],
                     sr=sr,
                     frame_length=PROSODY_SETTINGS["frame_length"],
                     hop_length=H)
    voiced_flag = ~np.isnan(f0)
    f0 = np.where(voiced_flag, f0, 0.0).astype(np.float32)
    voiced_flag = voiced_flag.astype(np.float32)

    # Stack only what the user enabled, in this order:
    bank = []
    for name in EXTRA_FEATURES:
        if name == "f0_hz": bank.append(f0)
        elif name == "voiced_flag": bank.append(voiced_flag)
        elif name == "rms": bank.append(rms)
        elif name == "spec_centroid": bank.append(spec_centroid)
        elif name == "spec_bandwidth": bank.append(spec_bandwidth)
        elif name == "spec_rolloff": bank.append(spec_rolloff)
        # elif name == "spec_flux": bank.append(spec_flux)
    F = np.vstack(bank).astype(np.float32) if bank else np.zeros((0, f0.shape[0]), np.float32)

    # Align length with MFCC frames (T_target)
    T = F.shape[1]
    if T < T_target:
        pad = np.repeat(F[:, -1:], T_target - T, axis=1)
        F = np.concatenate([F, pad], axis=1)
    elif T > T_target:
        F = F[:, :T_target]
    return F.T  # -> (T_target, num_extra)

# ---- Phase B1: extra per-frame features (pitch/energy/spectral) ----
def _safe_len_align(A, B, axis=0):
    """Pad/truncate B to match A along axis."""
    import numpy as np
    if A.shape[axis] == B.shape[axis]:
        return B
    T = A.shape[axis]
    if B.shape[axis] > T:
        slicer = [slice(None)] * B.ndim
        slicer[axis] = slice(0, T)
        return B[tuple(slicer)]
    # pad
    pad_width = [(0,0)] * B.ndim
    pad_width[axis] = (0, T - B.shape[axis])
    return np.pad(B, pad_width, mode='edge')

def _zscore_per_dim(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)

def _framewise_extras(y, sr, hop_length, n_fft):
    import numpy as np
    import librosa

    # Framewise RMS energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)  # [1, T]
    rms = rms.T  # [T, 1]

    # Spectral features (all framewise, shape [T, 1] each)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85).T

    # Spectral flux: L2 diff between consecutive normalized magnitude spectra
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))  # [F, T]
    S = S / (S.sum(axis=0, keepdims=True) + 1e-8)
    flux = np.sqrt(((np.diff(S, axis=1))**2).sum(axis=0, keepdims=True))  # [1, T-1]
    flux = np.concatenate([flux[:, :1], flux], axis=1).T  # [T, 1], pad first frame

    # Pitch contour using librosa.pyin (robust, returns Hz + unvoiced as nan)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                             frame_length=n_fft, hop_length=hop_length, sr=sr)
    # f0: [T] in Hz with NaNs for unvoiced → replace NaN by 0, keep a voiced mask
    f0_hz = np.nan_to_num(f0, nan=0.0)[:, None]  # [T,1]
    vuv   = (~np.isnan(f0)).astype(np.float32)[:, None]  # [T,1] voiced=1, unvoiced=0

    # Stack per-frame extras: [T, D_extra]
    extras = np.concatenate([rms, spec_centroid, spec_bw, spec_rolloff, flux, f0_hz, vuv], axis=1)
    return extras  # [T, 7]

def _concat_extras_per_frame(mfcc_stack, y, sr, hop_length, n_fft,
                             use_pitch=True, use_energy=True, use_spectral=True):
    """
    mfcc_stack: [T, D_mfcc]  (your current MFCC(+Δ,+ΔΔ) already stacked over D)
    returns:   [T, D_mfcc + D_extra]
    """
    import numpy as np
    D_extra = 0
    pieces = [mfcc_stack]

    if use_pitch or use_energy or use_spectral:
        extras = _framewise_extras(y, sr, hop_length, n_fft)  # [T_e, 7]
        # Option mask by groups
        idxs = []
        if use_energy:   idxs += [0]         # rms
        if use_spectral: idxs += [1,2,3,4]   # centroid, bw, rolloff, flux
        if use_pitch:    idxs += [5,6]       # f0_hz, vuv
        if idxs:
            extras = extras[:, idxs]         # [T_e, D_sel]
            extras = _safe_len_align(mfcc_stack, extras, axis=0)
            pieces.append(extras)
            D_extra = extras.shape[1]

    out = np.concatenate(pieces, axis=1)     # [T, D_mfcc + D_extra]
    return out, D_extra


if __name__ == "__main__":
    process_audio_files("train")
    process_audio_files("test")
