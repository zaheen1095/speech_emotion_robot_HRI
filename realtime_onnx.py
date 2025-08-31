import numpy as np
import sounddevice as sd
import onnxruntime as ort
from config import FEATURE_SETTINGS, CLASSES, RESPONSES, INFERENCE_SETTINGS, CALIBRATION_DIR
from extract_features import extract_mfcc
from pathlib import Path
import json

# --- Load ONNX Model ---
onnx_model_path = "models/model_c0.onnx"   # <-- use the exported file
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# optional: load calibrated threshold if present
sad_threshold = INFERENCE_SETTINGS["sad_threshold"]
thrfile = Path(CALIBRATION_DIR) / "threshold.txt"
try:
    if thrfile.exists():
        sad_threshold = float(open(thrfile, "r").read().strip())
        print(f"[C3.1] Using calibrated sad_threshold={sad_threshold:.2f}")
except Exception as e:
    print(f"[C3.1] Threshold default used ({e}).")

# --- Constants from config ---
DURATION = FEATURE_SETTINGS['max_duration']
SAMPLE_RATE = FEATURE_SETTINGS['sample_rate']
MAX_TIMESTEPS = FEATURE_SETTINGS['max_len']

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print(f"\nRecording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def predict_emotion(audio: np.ndarray):
    feats = extract_mfcc(array=audio, sr=SAMPLE_RATE)          # (T, D), trims/pads to MAX_TIMESTEPS
    inp = np.ascontiguousarray(feats[np.newaxis, :, :], dtype=np.float32)  # (1, T, D)

    logits = session.run(None, {input_name: inp})[0]           # (1, num_classes)
    p = _softmax(logits)[0]                                    # (num_classes,)
    p_happy, p_sad = float(p[0]), float(p[1])

    min_conf = INFERENCE_SETTINGS["min_confidence"]
    idx  = 1 if p_sad >= sad_threshold else 0
    conf = p[idx]

    emotion = "Uncertain" if conf < min_conf else CLASSES[idx]
    reply   = RESPONSES[emotion]
    return emotion, conf, reply

if __name__ == "__main__":
    try:
        while True:
            input("\nPress Enter to record (Ctrl+C to stop)...")
            audio = record_audio()
            emotion, conf, reply = predict_emotion(audio)
            print(f"\nPredicted Emotion: [{emotion.upper():8} {conf:.2f}] → {reply}")
    except KeyboardInterrupt:
        print("\nExiting.")
