import librosa
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from config import FEATURE_SETTINGS, CLASSES, RESPONSES
from extract_features import extract_mfcc

# --- Load ONNX Model ---
onnx_model_path = "models/best_model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

# --- Constants ---
DURATION = 3.0
SAMPLE_RATE = FEATURE_SETTINGS['sample_rate']
MAX_TIMESTEPS = 150

# --- Record Audio ---2
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print(f"\n Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

# --- Extract Features ---
# def extract_mfcc_from_array(y, sr):                       # Now I am not going to use it, to make the centralised the code .
#     mel_spec = librosa.feature.melspectrogram(
#         y=y,
#         sr=sr,
#         n_fft=FEATURE_SETTINGS['n_fft'],
#         hop_length=FEATURE_SETTINGS['hop_length'],
#         n_mels=FEATURE_SETTINGS['n_mels'],
#         fmin=FEATURE_SETTINGS['fmin'],
#         fmax=FEATURE_SETTINGS['fmax']
#     )
#     mel_db = librosa.power_to_db(mel_spec)
#     mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=FEATURE_SETTINGS['n_mfcc'])

#     features = [mfcc]
#     if FEATURE_SETTINGS['use_delta']:
#         features.append(librosa.feature.delta(mfcc))
#     if FEATURE_SETTINGS['use_delta_delta']:
#         features.append(librosa.feature.delta(mfcc, order=2))

#     return np.vstack(features).T

# --- Normalize Feature Length ---
def normalize_length(features, target_len=MAX_TIMESTEPS):
    if features.shape[0] < target_len:
        pad_amt = target_len - features.shape[0]
        features = np.pad(features, ((0, pad_amt), (0, 0)))
    else:
        features = features[:target_len, :]
    return features.astype(np.float32)

# --- Predict ---
# def predict_emotion(audio):
#     features = extract_mfcc_from_array(audio, SAMPLE_RATE)
#     features = normalize_length(features)
#     input_data = features[np.newaxis, :, :]  # shape (1, 150, 39)
#     outputs = session.run(None, {input_name: input_data})
#     predicted_idx = np.argmax(outputs[0])
#     return CLASSES[predicted_idx]


# ─── Predict ────────────────────────────────────────────────────────────
def predict_emotion(audio: np.ndarray):
    # 1) Denoise & MFCC → (MAX_LEN, n_mfcc)
    
    feats = extract_mfcc(array=audio, sr=SAMPLE_RATE)
    # 2) Batch dimension
    inp = feats[np.newaxis, :, :]
    # 3) ONNX inference (raw logits)
    raw = session.run(None, {input_name: inp})[0][0]
    # 4) Softmax to true probabilities
    exps= np.exp(raw - np.max(raw))
    probs= exps / exps.sum()
    # 5) Pick label + threshold
    idx, conf = int(np.argmax(probs)), float(np.max(probs))
    emotion = "Uncertain" if conf < 0.5 else CLASSES[idx]
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