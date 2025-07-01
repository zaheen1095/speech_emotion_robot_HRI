import sys
import numpy as np
import librosa
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
import pyttsx3
import threading

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
)
# your own config
from config import FEATURE_SETTINGS, CLASSES

# Choose backend: ONNX or PyTorch
USE_ONNX = True
responses = {
    "happy": "You sound happy! I'm glad to hear that.",
    "sad": "I'm here for you. It's okay to feel sad. Do you want to talk about it?",
    "Uncertain": "I am not sure how you are feeling. Would you like to try again."
}

# --- Load model once
if USE_ONNX:
    import onnxruntime as ort
    session = ort.InferenceSession("models/best_model.onnx")
    input_name = session.get_inputs()[0].name
else:
    import torch
    from models.cnn_bilstm import CNNBiLSTM
    model = CNNBiLSTM(input_dim=39, num_classes=len(CLASSES))
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
    model.eval()

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

class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(300)
        if is_user:
            self.setStyleSheet(
                "background:#e0e0e0; border-radius:10px; padding:8px;"
            )
        else:
            self.setStyleSheet(
                "background:#cfe9ff; border-radius:10px; padding:8px;"
            )


# --- GUI App
class EmotionApp(QWidget):
    def __init__(self):
        # ---- before chatbot code
        # super().__init__()
        # self.setWindowTitle("Speech Emotion Recognition (Live)")
        # self.setGeometry(100, 100, 400, 500)

        # self.label = QLabel("Click 'Record' to begin")
        # self.label.setAlignment(Qt.AlignCenter)

        # self.result_label = QLabel("")
        # self.result_label.setAlignment(Qt.AlignCenter)
        # self.result_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        # self.button = QPushButton("Record & Predict")
        # self.button.clicked.connect(self.record_and_predict)

        # layout = QVBoxLayout()
        # layout.addWidget(self.label)
        # layout.addWidget(self.result_label)
        # layout.addWidget(self.button)
        # self.setLayout(layout)
        # -------------------------------------

        super().__init__()
        self.setWindowTitle("Emotion Bot Chat")
        self.setGeometry(100, 100, 400, 500)

        # Chat area (vertical) inside a scrollable container
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        container = QWidget()
        container.setLayout(self.chat_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        # Record button
        self.button = QPushButton(" Record & Predict ...")
        self.button.clicked.connect(self.record_and_predict)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        main_layout.addWidget(self.button)

        # TTS engine
        self.engine = pyttsx3.init()
        self.is_processing = False
        # Keep reference to scroll for auto-scroll
        self.scroll_area = scroll

    def append_message(self, text, is_user=False):
        bubble = ChatBubble(text, is_user)
        hbox   = QHBoxLayout()
        if is_user:
            hbox.addStretch()
            hbox.addWidget(bubble)
        else:
            hbox.addWidget(bubble)
            hbox.addStretch()
        frame = QFrame()
        frame.setLayout(hbox)
        self.chat_layout.addWidget(frame)
        # auto-scroll to bottom
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finish(self):
        self.is_processing = False
        self.button.setEnabled(True)

    def _speak_and_finish(self, text):
        """ TTS on background thread, re-enable when done """
        self.engine.say(text)
        self.engine.runAndWait()
        self._finish()

    # def record_and_predict(self):
    #     try:
    #         self.label.setText("Recording for 3 seconds...")
            
    #         self.result_label.setText("")
    #         QApplication.processEvents()

    #         duration = 3.0
    #         sr = FEATURE_SETTINGS['sample_rate']
    #         audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    #         sd.wait()
    #         audio = audio.flatten()

    #         # --- Normalize amplitude
    #         if np.max(np.abs(audio)) > 0:
    #             audio = audio / np.max(np.abs(audio))

    #         features = extract_mfcc_from_array(audio, sr)
    #         input_data = features[np.newaxis, :, :]  # shape (1, 150, 39)

    #         if USE_ONNX:
    #             output = session.run(None, {input_name: input_data})  # output is a list
    #             probs = output[0][0]  # get first row (batch size 1), shape (2,)
    #             conf = np.max(probs)
    #             pred_idx = np.argmax(probs)

    #             if conf < 0.6:
    #                 emotion = "Uncertain"
    #             else:
    #                 emotion = CLASSES[pred_idx]
    #         else:
    #             input_tensor = torch.tensor(input_data).unsqueeze(0)
    #             with torch.no_grad():
    #                 output = model(input_tensor)
    #                 probs = output.numpy()[0]
    #                 conf = np.max(probs)
    #                 pred_idx = np.argmax(probs)
    #                 emotion = CLASSES[pred_idx] if conf >= 0.6 else "Uncertain"
    #         print(f"Predicted first: {emotion}, Confidence: {conf:.2f}, Probabilities: {probs}")

    #         # emotion = CLASSES[pred_idx]
    #         engine  = pyttsx3.init()
    #         engine.say(responses[emotion])
    #         engine.runAndWait()

    #         # --- Display in GUI
    #         self.label.setText("Done Recording") 
    #         self.result_label.setText(f"Detected Emotion: {emotion.upper()} ({conf: .2f})")
    #         print(f"Prediction second: {emotion} | Confidence: {conf:.2f} | Probabilities: {probs}")
    def record_and_predict(self):
        if self.is_processing:
            return
        self.is_processing = True
        self.button.setEnabled(False)
        # Indicate recording
        self.append_message("⏺️ Recording...", is_user=True)
        QApplication.processEvents()

        try:
            # 1) Record
            duration = 3.0
            sr       = FEATURE_SETTINGS['sample_rate']
            audio    = sd.rec(int(duration*sr), samplerate=sr, channels=1)
            sd.wait()
            audio    = audio.flatten()

            # if np.max(np.abs(audio))>0:
            #     audio = audio/np.max(np.abs(audio))
            #  check silence
            if np.max(np.abs(audio)) < 0.02:
                bot_reply = "I couldn't hear anything. Could you try speaking a bit louder?"
                self.append_message(bot_reply, is_user=False)
                threading.Thread(
                    target=self._speak_and_finish,
                    args=(bot_reply,),
                    daemon=True
                ).start()
                return

            
            # 2) Extract features
            feats = extract_mfcc_from_array(audio, sr)
            # inp   = feats[np.newaxis,:,:]
            inp   = feats[np.newaxis,:,:].astype(np.float32)

            # 3) Infer
            if USE_ONNX:
                probs = session.run(None, {input_name: inp})[0][0]
            else:
                import torch
                tensor = torch.tensor(inp).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            conf     = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            emotion  = "Uncertain" if conf<0.6 else CLASSES[pred_idx]
            # response = responses.get(emotion, "Could you say that again?")

            # # 4) Show bot reply
            # self.append_message(response, is_user=False)
            bot_reply = responses[emotion]

            # 4) display
            self.append_message(bot_reply, is_user=False)

            # 5) Speak off-thread
            threading.Thread(
                target=lambda: (self.engine.say(bot_reply), self.engine.runAndWait()),
                daemon=True
            ).start()

        # except Exception as e:
        #     err = f"Error: {e}"
        #     self.append_message(err, is_user=False)
        #     print(err)

            # 5) speak in background
            threading.Thread(
                target=self._speak_and_finish,
                args=(bot_reply,),
                daemon=True
            ).start()

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
