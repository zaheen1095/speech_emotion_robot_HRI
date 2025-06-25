import numpy as np

def pad_sequence(seq, max_len=300):
    """Pads or trims a sequence to fixed length."""
    if len(seq) > max_len:
        return seq[:max_len, :]
    pad_width = max_len - len(seq)
    return np.pad(seq, ((0, pad_width), (0, 0)), mode='constant')