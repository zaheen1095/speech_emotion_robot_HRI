from pathlib import Path

# === Paths (using pathlib) ===
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'datasets'
RAW_AUDIO_DIR = DATASET_DIR / 'raw_audio'
RESAMPLED_DIR = DATASET_DIR / 'resampled_audio'
FEATURES_DIR = DATASET_DIR / 'features'
MODEL_DIR = BASE_DIR / 'models'

# === Feature Extraction ===
FEATURE_SETTINGS = {
    "sample_rate": 16000,
    "n_fft": 512,         # etc…
    "hop_length": 256,
    "n_mels": 40,
    "fmin": 20,
    "fmax": 8000,
    "n_mfcc": 13,
    "use_delta": True,
    "use_delta_delta": True
}

# === Training Defaults ===
CLASSES = ['happy', 'sad']
BATCH_SIZE = 32