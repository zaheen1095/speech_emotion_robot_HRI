import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import FEATURE_SETTINGS, RESAMPLED_DIR, FEATURES_DIR

def extract_mfcc(audio_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(
            audio_path,
            sr=FEATURE_SETTINGS['sample_rate'],
            duration=3.0
        )

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

        features = [mfcc]
        if FEATURE_SETTINGS['use_delta']:
            features.append(librosa.feature.delta(mfcc))
        if FEATURE_SETTINGS['use_delta_delta']:
            features.append(librosa.feature.delta(mfcc, order=2))

        return np.vstack(features).T

    except Exception as e:
        raise ValueError(f"Error processing {audio_path}: {str(e)}")

def normalize_length(features, target_len=150):
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
        os.makedirs(output_dir / emotion, exist_ok=True)

        for filename in os.listdir(emotion_dir):
            if filename.endswith(".wav"):
                input_path = emotion_dir / filename
                output_path = output_dir / emotion / f"{Path(filename).stem}.npy"

                try:
                    features = extract_mfcc(str(input_path))
                    features = normalize_length(features)
                    np.save(output_path, features)
                except ValueError as e:
                    print(f" Skipping {filename}: {str(e)}")

if __name__ == "__main__":
    process_audio_files("train")
    process_audio_files("test")
