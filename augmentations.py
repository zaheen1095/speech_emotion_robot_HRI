# augmentations.py
import numpy as np

# --------- Feature-level (SpecAugment-style) ---------
def spec_augment(
    feats: np.ndarray,
    time_mask_pct: float = 0.10,   # up to 10% of time axis
    freq_mask_pct: float = 0.10,   # up to 10% of freq axis
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
    p: float = 0.5,
) -> np.ndarray:
    """
    Apply simple SpecAugment to a [C, F, T] or [F, T] array (MFCC/mel).
    - Masks random contiguous time and frequency bands.
    - No augmentation if random > p.
    """
    if np.random.rand() > p:
        return feats

    x = feats.copy()
    if x.ndim == 2:
        # [F, T] -> add channel dim
        x = x[None, :, :]
        squeeze_back = True
    else:
        squeeze_back = False

    _, F, T = x.shape

    # freq masks
    max_f = max(1, int(freq_mask_pct * F))
    for _ in range(num_freq_masks):
        width = np.random.randint(1, max_f + 1)
        f0 = np.random.randint(0, max(1, F - width + 1))
        x[:, f0:f0 + width, :] = 0.0

    # time masks
    max_t = max(1, int(time_mask_pct * T))
    for _ in range(num_time_masks):
        width = np.random.randint(1, max_t + 1)
        t0 = np.random.randint(0, max(1, T - width + 1))
        x[:, :, t0:t0 + width] = 0.0

    if squeeze_back:
        x = x[0]
    return x


# --------- Waveform-level (optional, train-only) ---------
# Will use audiomentations if available; otherwise, fall back to no-op.
try:
    from audiomentations import (
        Compose, AddGaussianNoise, Gain, PitchShift, TimeStretch,
        BandPassFilter, ClippingDistortion, Shift
    )

    def build_wave_augmenter(sample_rate: int):
        return Compose([
            AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.01, p=0.35),
            BandPassFilter(min_center_freq=200, max_center_freq=3800, p=0.30),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.25),
            PitchShift(min_semitones=-2, max_semitones=+2, p=0.25),
            Gain(min_gain_in_db=-6, max_gain_in_db=+6, p=0.25),
            ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=20, p=0.20),
            Shift(min_fraction=-0.1, max_fraction=0.1, p=0.25),
        ])

    def augment_waveform(y: np.ndarray, sr: int, augmenter=None) -> np.ndarray:
        if augmenter is None:
            augmenter = build_wave_augmenter(sr)
        return augmenter(samples=y.astype(np.float32), sample_rate=sr).astype(np.float32)

except Exception:
    # Fallback: no-op if audiomentations is not installed
    def build_wave_augmenter(sample_rate: int):
        return None
    def augment_waveform(y: np.ndarray, sr: int, augmenter=None) -> np.ndarray:
        return y
