import os, numpy as np, soundfile as sf, tqdm
from extract_features import load_filelist  # or your util
from ssl_frontend import SSLFrontend
import config as cfg

os.makedirs(cfg.SSL_CACHE_DIR, exist_ok=True)
ssl = SSLFrontend(cfg.SSL_MODEL, cfg.SSL_FREEZE)

for wav_path, label in tqdm.tqdm(load_filelist("file_order.txt")):  # adapt to your index
    y, sr = sf.read(wav_path)
    # ensure mono/16k
    if sr != 16000: 
        # reuse your resample util here
        from utils import resample_to_16k
        y = resample_to_16k(y, sr)
    if y.ndim > 1: y = y.mean(axis=1)
    feats = ssl(y.astype(np.float32))
    out = os.path.join(cfg.SSL_CACHE_DIR, os.path.basename(wav_path).replace(".wav","_ssl.npy"))
    np.save(out, feats)
