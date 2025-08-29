# resample_audio.py
import librosa, soundfile as sf
from pathlib import Path
from config import AUGMENTED_DIR, RESAMPLED_DIR, FEATURE_SETTINGS, RAW_AUDIO_DIR

TARGET_SR = FEATURE_SETTINGS["sample_rate"]

def resample_file(in_path: Path, out_path: Path):
    try:
        y, sr = librosa.load(str(in_path), sr=None, mono=True)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), y, sr)
    except Exception as e:
        print(f"[skip] {in_path}: {e}")

def process_split(split="train"):
    # in_root  = AUGMENTED_DIR / split
    in_root = RAW_AUDIO_DIR / split
    out_root = RESAMPLED_DIR / split
    
    wav_exts = (".wav", ".WAV")
    files = [p for p in in_root.rglob("*") if p.suffix in wav_exts]
    if not files:
        print(f"[warn] No WAVs under {in_root} — put your original audio there.?")
        return

    print(f"[info] Resampling {len(files)} files to {TARGET_SR} Hz")
    for in_path in files:
        rel = in_path.relative_to(in_root)              # keep tree
        out_path = out_root / rel
        resample_file(in_path, out_path)

if __name__ == "__main__":
    process_split("train")
    process_split("test")   # test has only originals (no aug), still resampled
    print("[done] Resampled WAVs in datasets/resampled_audio/")
