import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import FEATURE_SETTINGS, RESAMPLED_DIR, FEATURES_DIR

def extract_mfcc(audio_path: str) -> np.ndarray:
    """
    Extract MFCC features (+ delta, delta-delta) from an audio file.
    Applies Mel spectrogram with fmin/fmax and converts to MFCCs.
    """
    try:
        # Load and trim/pad audio to fixed length
        y, sr = librosa.load(
            audio_path,
            sr=FEATURE_SETTINGS['sample_rate'],
            duration=3.0  # Ensures consistent feature dimensions
        )

        # Step 1: Compute Mel spectrogram with custom fmin/fmax
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=FEATURE_SETTINGS['n_fft'],
            hop_length=FEATURE_SETTINGS['hop_length'],
            n_mels=FEATURE_SETTINGS['n_mels'],
            fmin=FEATURE_SETTINGS['fmin'],
            fmax=FEATURE_SETTINGS['fmax']
        )

        # Step 2: Convert to log-scaled dB
        mel_db = librosa.power_to_db(mel_spec)

        # Step 3: Convert Mel spectrogram to MFCC
        mfcc = librosa.feature.mfcc(
            S=mel_db,
            n_mfcc=FEATURE_SETTINGS['n_mfcc']
        )

        # Step 4: Stack delta and delta-delta if enabled
        features = [mfcc]
        if FEATURE_SETTINGS['use_delta']:
            features.append(librosa.feature.delta(mfcc))
        if FEATURE_SETTINGS['use_delta_delta']:
            features.append(librosa.feature.delta(mfcc, order=2))

        return np.vstack(features).T  # Shape: (time_steps, features)

    except Exception as e:
        raise ValueError(f"Error processing {audio_path}: {str(e)}")

def process_audio_files(split: str):
    """
    Process all audio files in a split (train/test) and save features as .npy.
    """
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
                    np.save(output_path, features)
                except ValueError as e:
                    print(f" Skipping {filename}: {str(e)}")

if __name__ == "__main__":
    process_audio_files("train")
    process_audio_files("test")
