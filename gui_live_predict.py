import sys
import numpy as np
import librosa
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

# Choose backend: ONNX or PyTorch
USE_ONNX = True

# --- Load model
if USE_ONNX:
    import onnxruntime as ort
    session = ort.InferenceSession("models/best_model.onnx")
    input_name = session.get_inputs()[0].name
else:
    import torch
    from models.cnn_bilstm import CNNBiLSTM
    from config import CLASSES
    model = CNNBiLSTM(input_dim=39, num_classes=2)
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
    model.eval()

# --- Feature Extraction Settings
from config import FEATURE_SETTINGS, CLASSES

def extract_mfcc_from_array(y, sr, max_len=150):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=FEATURE_SETTINGS['n_fft'],
        hop_length=FEATURE_SETTINGS['hop_length'],
        n_mels=FEATURE_SETTINGS['n_mels'],
        fmin=FEATURE_SETTINGS['fmin'],
        fmax=FEATURE_SETTINGS['fmax']
    )
    mel_db = librosa.power_to_db(mel_spec)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=FEATURE_SETTINGS['n_mfcc'])

    features = [mfcc]
    
    if FEATURE_SETTINGS['use_delta']:
        features.append(librosa.feature.delta(mfcc))
    if FEATURE_SETTINGS['use_delta_delta']:
        features.append(librosa.feature.delta(mfcc, order=2))

    stacked = np.vstack(features).T

    if stacked.shape[0] < max_len:
        pad = max_len - stacked.shape[0]
        stacked = np.pad(stacked, ((0, pad), (0, 0)))
    else:
        stacked = stacked[:max_len, :]
    
    print("✅ MFCC shape (after stack):", stacked.shape)

    return stacked.astype(np.float32)



# --- GUI App
class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Emotion Recognition (Live)")
        self.setGeometry(200, 200, 400, 200)

        self.label = QLabel("🎙️ Click 'Record' to begin")
        self.label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.button = QPushButton("Record & Predict")
        self.button.clicked.connect(self.record_and_predict)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def record_and_predict(self):
        try:
            self.label.setText("Recording for 3 seconds...")
            
            self.result_label.setText("")
            QApplication.processEvents()

            duration = 3.0
            sr = FEATURE_SETTINGS['sample_rate']
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
            sd.wait()
            audio = audio.flatten()

            features = extract_mfcc_from_array(audio, sr)
            input_data = features[np.newaxis, :, :]  # shape (1, 150, 39)

            # if USE_ONNX:
            #     output = session.run(None, {input_name: input_data})[0]
            #     probs = output[0]  # shape: [2]
            #     conf = np.max(probs)
            #     pred_idx = np.argmax(output)
            #     if conf < 0.6:
            #         emotion = "Uncertain"
            #     else:
            #         emotion = CLASSES[pred_idx]
            # else:
            #     input_tensor = torch.tensor(input_data).unsqueeze(0)
            #     with torch.no_grad():
            #         output = model(input_tensor)
            #         pred_idx = torch.argmax(output, dim=1).item()
            if USE_ONNX:
                output = session.run(None, {input_name: input_data})  # output is a list
                probs = output[0][0]  # get first row (batch size 1), shape (2,)
                conf = np.max(probs)
                pred_idx = np.argmax(probs)

                if conf < 0.6:
                    emotion = "Uncertain"
                else:
                    emotion = CLASSES[pred_idx]
            else:
                input_tensor = torch.tensor(input_data).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
            print(f"Predicted first: {emotion}, Confidence: {conf:.2f}, Probabilities: {probs}")

            emotion = CLASSES[pred_idx]
            self.label.setText("Done Recording")
            print(f"Prediction second: {emotion}, Confidence: {conf:.2f}")
            self.result_label.setText(f"Detected Emotion: {emotion.upper()}")
        except Exception as e:
            self.label.setText("Error - Click 'Record' to retry")
            self.result_label.setText(f"Error: {str(e)}")
            print(f"Error: {e}")
            
        print(f"🎯 You said it with a {emotion.upper()} tone.")

# --- Run
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
