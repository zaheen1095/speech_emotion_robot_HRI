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
    "use_delta_delta": True,
    "max_len": 150,          # pad/truncate frames
    "max_duration": 3.0      # seconds to load
}

# === Training Defaults ===
CLASSES = ['happy', 'sad']
BATCH_SIZE = 32
RESPONSES = {
    "happy": "You sound happy! I'm glad to hear that.",
    "sad": "I'm here for you. It's okay to feel sad. Do you want to talk about it?",
    "Uncertain": "I am not sure how you are feeling. Would you like to try again."
}