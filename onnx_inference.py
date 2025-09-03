import numpy as np
import sounddevice as sd
import onnxruntime as ort
from config import FEATURE_SETTINGS, CLASSES, RESPONSES, INFERENCE_SETTINGS
from extract_features import extract_mfcc

# --- Load ONNX Model ---
# onnx_model_path = "models/best_model.onnx"
onnx_model_path = "models/model_c0.onnx" 
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# --- Constants from config ---
DURATION = FEATURE_SETTINGS['max_duration']     # 5.0
SAMPLE_RATE = FEATURE_SETTINGS['sample_rate']   # 16000
MAX_TIMESTEPS = FEATURE_SETTINGS['max_len']     # 313

# --- Record Audio ---
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print(f"\n Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# --- Predict ---
def predict_emotion(audio: np.ndarray):
    # Centralized feature extraction (already pads/trims to MAX_TIMESTEPS)
    feats = extract_mfcc(array=audio, sr=SAMPLE_RATE)      # shape (MAX_TIMESTEPS, 39)
    input_data = np.ascontiguousarray(feats[np.newaxis, :, :],   # (1, T, 39)
                                      dtype=np.float32)                  # shape (1, T, 39)

    # ONNX forward → logits
    outputs = session.run(None, {input_name: input_data})
    logits = outputs[0]                                    # shape (1, num_classes)

    # Softmax → probabilities (numerically stable), keep your 'p' name
    # exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # p = (exps / np.sum(exps, axis=1, keepdims=True))[0]    # shape (num_classes,)
    # p_happy, p_sad = float(p[0]), float(p[1])
    p = _softmax(logits)[0]                                      # (num_classes,)
    p_happy, p_sad = float(p[0]), float(p[1])

    th_sad  = INFERENCE_SETTINGS["sad_threshold"]
    min_conf = INFERENCE_SETTINGS["min_confidence"]

    # Prefer 'sad' once it crosses the threshold; otherwise 'happy'
    idx  = 1 if p_sad >= th_sad else 0
    conf = p[idx]

    emotion = "Uncertain" if conf < min_conf else CLASSES[idx]
    reply   = RESPONSES[emotion]
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
