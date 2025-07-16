import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from config import FEATURE_SETTINGS, RESAMPLED_DIR, FEATURES_DIR
import noisereduce as nr

def extract_mfcc(audio_path: str = None, array:np.ndarray = None, sr:int = None) -> np.ndarray:
    try:
        if audio_path is not None:
            y, sr = librosa.load(
                audio_path,
                sr=FEATURE_SETTINGS['sample_rate'],
                duration= FEATURE_SETTINGS.get('max_duration', 3.0)
            )
        else :
            assert array is not None and sr == FEATURE_SETTINGS['sample_rate'], \
            "When not using audio_path, you must pass array and matching sr"
            y = array


        #-------- Noise reduction -------- #
        noise_clip = y[: int(0.5 * sr)]
        y = nr.reduce_noise(
            y=y,                
            sr=sr,              
            y_noise=noise_clip, 
            n_fft=FEATURE_SETTINGS['n_fft'],
            hop_length=FEATURE_SETTINGS['hop_length']
        )

        # **NEW**: ensure no NaNs/Infs slip through 
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)       # for the checking NAN in the dataset.

        #------ end noise reduction code -------#

        #--- Mel-spectogram -> MFCC
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
        if FEATURE_SETTINGS.get('use_delta', False):
            features.append(librosa.feature.delta(mfcc))
        if FEATURE_SETTINGS.get('use_delta_delta', False):
            features.append(librosa.feature.delta(mfcc, order=2))
        
        stacked = np.vstack(features).T
        #-- end MFCC 

        # ─── Pad or truncate to fixed length ────────────────────────
        max_len = FEATURE_SETTINGS.get('max_len', 150)
        if stacked.shape[0] < max_len:
            pad = max_len - stacked.shape[0]
            stacked = np.pad(stacked, ((0, pad), (0, 0)))
        else:
            stacked = stacked[:max_len, :]
        # ─── End shape normalization ────────────────────────────────

        return stacked.astype(np.float32)

        # return np.vstack(features).T

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
        os.makedirs(output_dir / emotion, exist_ok=True)

        for filename in os.listdir(emotion_dir):
            if filename.endswith(".wav"):
                input_path = emotion_dir / filename
                output_path = output_dir / emotion / f"{Path(filename).stem}.npy"

                try:
                    features = extract_mfcc(audio_path=str(input_path))
                    # features = extract_mfcc(str(input_path))
                    # features = normalize_length(features)
                    np.save(output_path, features)
                except ValueError as e:
                    print(f" Skipping {filename}: {str(e)}")

if __name__ == "__main__":
    process_audio_files("train")
    process_audio_files("test")
