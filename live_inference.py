import librosa
import numpy as np
import torch
import sounddevice as sd
from config import FEATURE_SETTINGS, CLASSES
from models.cnn_bilstm import CNNBiLSTM
from extract_features import extract_mfcc

# --- Load Trained PyTorch Model ---
model = CNNBiLSTM(input_dim=39, num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device('cpu')))
model.eval()

# --- Record Audio ---
def record_audio(duration=3, sample_rate=16000):
    print(f"\n🎙️ Speak now (recording for {duration} seconds)...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

# --- Extract MFCC Features ---
# def extract_features(audio, sample_rate):
#     mel_spec = librosa.feature.melspectrogram(
#         y=audio, sr=sample_rate,
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

#     combined = np.vstack(features).T

#     # Pad/trim to 150 timesteps
#     max_len = 150
#     if combined.shape[0] < max_len:
#         pad_amt = max_len - combined.shape[0]
#         combined = np.pad(combined, ((0, pad_amt), (0, 0)))
#     else:
#         combined = combined[:max_len, :]

#     return torch.FloatTensor(combined).unsqueeze(0)

# --- Predict Emotion ---
# def predict_emotion(audio):
#     features = extract_features(audio, FEATURE_SETTINGS['sample_rate'])
#     with torch.no_grad():
#         outputs = model(features)
#         _, predicted = torch.max(outputs, 1)
#     return CLASSES[predicted.item()]

def predict_emotion(audio):
    features = extract_mfcc(array=audio, sr=FEATURE_SETTINGS['sample_rate'])
    input_tensor = torch.tensor(features).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASSES[predicted.item()]


# --- Run ---
if __name__ == "__main__":
    try:
        while True:
            input("\nPress Enter to record (Ctrl+C to stop)...")
            audio = record_audio()
            emotion = predict_emotion(audio)
            print(f"\nPredicted Emotion: {emotion.upper()}")
    except KeyboardInterrupt:
        print("\nStopped.")
