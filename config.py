from pathlib import Path

# === Paths (using pathlib) ===
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'datasets'
RAW_AUDIO_DIR = DATASET_DIR / 'raw_audio'
RESAMPLED_DIR = DATASET_DIR / 'resampled_audio'
AUGMENTED_DIR   = DATASET_DIR / 'augmented_audio'

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
    "max_duration": 5.0,      # seconds to load
    "trim_db": 40
}

# === Training Defaults ===
CLASSES = ['happy', 'sad']
BATCH_SIZE = 32
RESPONSES = {
    "happy": "You sound happy! I'm glad to hear that.",
    "sad": "I'm here for you. It's okay to feel sad. Do you want to talk about it?",
    "Uncertain": "I am not sure how you are feeling. Would you like to try again."
}
CLASS_WEIGHTS = [1.0, 1.4]          # [happy, sad]
LABEL_SMOOTHING = 0.0
MONITOR_METRIC = "f1_score_macro"       # checkpoint/early-stop on this

INFERENCE_SETTINGS = {
    "sad_threshold": 0.40,          # prefer 'sad' when P(sad) >= threshold
    "min_confidence": 0.50          # below this, return "Uncertain"
}
AUG_PER_FILE = {"train": 1, "test": 0}               # change '1' to 2/3/etc. if you want more
AUG_SEED = 1337
SAD_THRESHOLD_OVERRIDE = None

# === Phase B switches ===
USE_EXTRA_FEATURES = True          # turn B1 on/off
USE_ATTENTION = True              # add attention later (B2) if gap persists

# Pitch/energy settings (aligned with MFCC hop/frames)
PROSODY_SETTINGS = {
    "f0_method": "yin",            # "yin" or "pyin"
    "frame_length": 1024,          # keep consistent with FEATURE_SETTINGS n_fft
    "hop_length": FEATURE_SETTINGS["hop_length"],
    "fmin_hz": 50,
    "fmax_hz": 500,
    "rolloff_percent": 0.85
}

# Which extra per-frame features to append (order matters)
EXTRA_FEATURES = [
    "f0_hz", "voiced_flag", "rms",
    "spec_centroid", "spec_bandwidth", "spec_rolloff"
   
]
# -------- Phase B3: eval-time augmentation (val/test) --------
USE_AUG_ON_VAL_TEST = True          # <— turn ON
VAL_TEST_AUG_PROB   = 0.50          # 50% of files get one augmentation
VAL_TEST_AUG_CHAIN  = False         # if True, can apply more than one

# individual op probs (used only if selected)
VAL_TEST_NOISE_PROB = 0.60
VAL_TEST_PITCH_PROB = 0.35
VAL_TEST_TIME_PROB  = 0.35
VAL_TEST_REVERB_PROB= 0.35

# parameter ranges
VAL_TEST_NOISE_SNR_DB = [0, 5, 10]     # pick one at random
VAL_TEST_PITCH_STEPS  = [-2, -1, 1, 2] # semitones
VAL_TEST_TIME_RANGE   = [0.90, 1.10]   # time stretch
VAL_TEST_IR_PRESET    = "small_room"   # if you support a convolution reverb

DATASET_PREFIXES = {
    "crema": "CREMA-D","cremad": "CREMA-D",  "CREMAD": "CREMA-D",
    "CREMA":  "CREMA-D",
    "ravdess": "RAVDESS",
    "tess": "TESS",
    "savee": "SAVEE",
    "jl": "JL",
    "iemocap": "IEMOCAP",
    "jlcorpus": "JL",
    "jl-corpus": "JL",
    "JLCORPUS" : "JL",
    "iemocap": "IEMOCAP",
    "creamad": "CREMA-D",
}

# Defaults (overridable by CLI)
TRAIN_DATASETS = ["IEMOCAP", "CREMA-D", "JL", "RAVDESS", "SAVEE", "TESS"]
# config.py  (add near TRAIN_DATASETS / HELDOUT_DATASETS)
DEFAULT_TEST_DATASETS = ["CREMA-D", "IEMOCAP", "JL", "RAVDESS", "TESS"]

HELDOUT_DATASETS = [] 

# ---- Phase C toggles ----
USE_SSL_FRONTEND = True          # set True to enable wav2vec2/HubERT
SSL_MODEL = "wav2vec2-base"       # or "hubert-base"
SSL_FREEZE = True                 # start frozen
SSL_FRAME_HOP_MS = 20             # for downstream temporal stride
SSL_CACHE_DIR = "ssl_cache"       # where *_ssl.npy will be written
