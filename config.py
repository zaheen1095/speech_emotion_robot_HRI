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
    "n_mels": 64,
    "fmin": 50,
    "fmax": 8000,
    "n_mfcc": 13,
    "use_delta": True,
    "use_delta_delta": True,
    "max_len": 313,          # pad/truncate frames
    "max_duration": 5.0      # seconds to load
}

# === Training Defaults ===
CLASSES = ['happy', 'sad']
BATCH_SIZE = 32
RESPONSES = {
    "happy": "You sound happy! I'm glad to hear that.",
    "sad": "I'm here for you. It's okay to feel sad. Do you want to talk about it?",
    "Uncertain": "I am not sure how you are feeling. Would you like to try again."
}
CLASS_WEIGHTS = [1.0, 1.1]          # [happy, sad]
LABEL_SMOOTHING = 0.0
MONITOR_METRIC = "recall_sad"       # checkpoint/early-stop on this

INFERENCE_SETTINGS = {
    "sad_threshold": 0.45,          # prefer 'sad' when P(sad) >= threshold
    "min_confidence": 0.50          # below this, return "Uncertain"
}