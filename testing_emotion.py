import torch
import numpy as np
import sys
from extract_features import extract_mfcc
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURE_SETTINGS, CLASSES

# Path to your test .wav file (example)
# audio_path = "datasets/raw_audio/test/happy/crema_happy_male46_actor87_IEO_LO.wav" # giving wrong emotion (happy: 0.08,sad: 0.92)
# audio_path = "datasets/raw_audio/test/happy/crema_happy_female36_actor75_ITS_XX.wav"  #giving correct emotion(happy: 0.60,sad: 0.40)

# audio_path = "datasets/raw_audio/test/happy/crema_happy_female36_actor75_IEO_MD.wav" #wrong (happy: 0.24,sad: 0.76)

# audio_path = "datasets/raw_audio/test/happy/jl_happy_male2-actor4_6b_1.wav"  #correct emotion (happy: 0.99,sad: 0.01)
# audio_path = "datasets/raw_audio/test/happy/ravdess-happy-female11-actor22-01-01-02.wav" #wrong emotion (happy: 0.49,sad: 0.51)

# audio_path = "datasets/raw_audio/test/happy/savee_happy_male3-actor3_h01.wav"  correct emotion (happy: 0.95,sad: 0.05)
audio_path = "datasets/raw_audio/test/happy/jl_happy_female2-actor2_10a_1.wav"  #correct emotion (happy=0.59, sad = 0.41)

# Load features from the given audio file
features = extract_mfcc(audio_path=audio_path)
input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape: [1, seq_len, input_dim]

# Load the trained model
model = CNNBiLSTM(
    input_dim=FEATURE_SETTINGS['n_mfcc'] * (
        1 + FEATURE_SETTINGS['use_delta'] + FEATURE_SETTINGS['use_delta_delta']
    ),
    num_classes=len(CLASSES)
)

model.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device('cpu')))
model.eval()

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_index].item()



for i, prob in enumerate(probabilities[0]):
    print(f"{CLASSES[i]}: {prob:.2f}")

print(f"\n Audio file: {audio_path}")
print(f" Predicted emotion: {CLASSES[predicted_index]}")
print(f" Confidence score: {confidence:.2f}")
