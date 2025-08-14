import numpy as np
from config import FEATURE_SETTINGS

def pad_sequence(seq, max_len: int = None):
    """Pads or trims a (T,F) sequence to fixed length."""
    if max_len is None:
        max_len = FEATURE_SETTINGS['max_len']
    T = len(seq)
    if T > max_len:
        return seq[:max_len, :]
    pad = max_len - T
    return np.pad(seq, ((0, pad), (0, 0)), mode='constant')
