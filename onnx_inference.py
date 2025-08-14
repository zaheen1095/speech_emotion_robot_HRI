import librosa
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from config import FEATURE_SETTINGS, CLASSES, RESPONSES, INFERENCE_SETTINGS
from extract_features import extract_mfcc

# --- Load ONNX Model ---
onnx_model_path = "models/best_model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

# --- Constants (match config) ---
DURATION = FEATURE_SETTINGS['max_duration']
SAMPLE_RATE = FEATURE_SETTINGS['sample_rate']
MAX_TIMESTEPS = FEATURE_SETTINGS['max_len']

# --- Record Audio ---
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print(f"\n Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

# --- Normalize Feature Length (kept for compatibility; extractor already pads) ---
def normalize_length(features, target_len=MAX_TIMESTEPS):
    if features.shape[0] < target_len:
        pad_amt = target_len - features.shape[0]
        features = np.pad(features, ((0, pad_amt), (0, 0)))
    else:
        features = features[:target_len, :]
    return features.astype(np.float32)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ─── Predict ────────────────────────────────────────────────────────────
def predict_emotion(audio: np.ndarray):
    # 1) MFCC via central extractor (pads/normalizes per config)
    feats = extract_mfcc(audio_path=None, array=audio, sr=SAMPLE_RATE)  # (MAX_TIMESTEPS, 39)
    input_data = feats[np.newaxis, :, :]  # (1, T, F)

    # 2) ONNX forward -> logits -> probabilities
    outputs = session.run(None, {input_name: input_data})
    logits = outputs[0].squeeze()          # shape (num_classes,)
    p = _softmax(logits).astype(float)     # numpy float probs
    p_happy, p_sad = float(p[0]), float(p[1])

    # 3) Decision with thresholds from config
    th_sad = INFERENCE_SETTINGS["sad_threshold"]
    min_conf = INFERENCE_SETTINGS["min_confidence"]

    idx = 1 if p_sad >= th_sad else 0            # prefer 'sad' once threshold crosses
    conf = float(p[idx])

    emotion = "Uncertain" if conf < min_conf else CLASSES[idx]
    reply = RESPONSES[emotion]
    return emotion, conf, reply

# --- Run ---
if __name__ == "__main__":
    try:
        while True:
            input("\n Press Enter to record (Ctrl+C to stop)...")
            audio = record_audio()
            emotion, conf, reply = predict_emotion(audio)
            print(f"\n Predicted Emotion: [{emotion.upper():8} {conf:.2f}] → {reply}")
    except KeyboardInterrupt:
        print("\n Exiting.")
