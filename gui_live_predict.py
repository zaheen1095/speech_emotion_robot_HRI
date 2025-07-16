import sys
import numpy as np
import librosa
import sounddevice as sd
import pyttsx3
import threading
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
)
# your own config
from config import FEATURE_SETTINGS, CLASSES, RESPONSES
from extract_features import extract_mfcc

# Choose backend: ONNX or PyTorch
USE_ONNX = True


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

# def extract_mfcc_from_array(y, sr, max_len=150):                  #I am not going to use now, later will check
#     mel_spec = librosa.feature.melspectrogram(
#         y=y, sr=sr,
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

#     stacked = np.vstack(features).T

#     if stacked.shape[0] < max_len:
#         pad = max_len - stacked.shape[0]
#         stacked = np.pad(stacked, ((0, pad), (0, 0)))
#     else:
#         stacked = stacked[:max_len, :]
    
#     print("MFCC shape (after stack):", stacked.shape)

#     return stacked.astype(np.float32)

class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(300)
        color = "#e0e0e0" if is_user else "#95abbe"
        self.setStyleSheet(f"background:{color}; border-radius:10px;padding:8px;")
        # if is_user:
        #     self.setStyleSheet(
        #         "background:#e0e0e0; border-radius:10px; padding:8px;"
        #     )
        # else:
        #     self.setStyleSheet(
        #         "background:#cfe9ff; border-radius:10px; padding:8px;"
        #     )


# --- GUI App
class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Emotion Detection Application")
        self.setGeometry(100, 100, 400, 500)

        # — Add a microphone indicator —
        self.mic_icon = QLabel()
        # load two small images (mic_off.png and mic_on.png) into your project folder
        self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio)
        self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio)
        self.mic_icon.setPixmap(self.mic_off)

        # Chat area
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        container = QWidget(); 
        container.setLayout(self.chat_layout)
        scroll    = QScrollArea(); 
        scroll.setWidgetResizable(True); scroll.setWidget(container)
        self.scroll_area   = scroll  #check this and below line code after analysing UI.
        self.is_processing = False

        # Record button
        self.button = QPushButton(" Record & Detect")
        self.button.setIcon(QIcon(self.mic_off))
        self.button.setIconSize(QSize(24,24))
        self.button.clicked.connect(self.record_and_predict)

        # Main layout: MIC ICON goes *before* the scroll
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
        main_layout.addWidget(scroll)
        main_layout.addWidget(self.button)

        # Record button with mic icons
        #ToDo - check the above UI is fine then delete this , otherwise check this UI is working accordingly or not.
        # self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio)
        # self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio)
        # self.button = QPushButton("Record & Detect")
        # self.button.setIcon(QIcon(self.mic_off))
        # self.button.setIconSize(QSize(24,24))
        # self.button.clicked.connect(self.record_and_predict)

        # # Layout
        # main = QVBoxLayout(self)
        # main.addWidget(scroll)
        # main.addWidget(self.button)

        # self.scroll_area   = scroll
        # self.is_processing = False

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
        """Turn mic icon off and re-enable the button."""
        self.mic_icon.setPixmap(self.mic_off)
        self.button.setText("🎤 Record & Detect")
        self.button.setIcon(QIcon(self.mic_off))
        self.button.setEnabled(True)
        self.is_processing = False
        #Todo, check above code is workign fine on UI keep it otherwise replace with the following one 
        # self.button.setEnabled(True)
        # self.button.setText("🎤 Record & Detect")
        # self.button.setIcon(QIcon(self.mic_off))
        # self.is_processing = False

    def _speak_and_finish(self, text):
        """Speak text (fresh engine) then call _finish()."""
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        self._finish()
    
    def record_and_predict(self):
        # 1) Guard against re-entry
        if self.is_processing:
            return
        self.is_processing = True
        self.button.setEnabled(False)

        # 2) Show “recording” state
        self.button.setText("🔴 Recording…")
        self.button.setIcon(QIcon(self.mic_on))
        QApplication.processEvents()

        try:
            # 3) Capture 3s of audio
            sr    = FEATURE_SETTINGS['sample_rate']
            audio = sd.rec(int(3 * sr), samplerate=sr, channels=1)
            sd.wait()
            audio = audio.flatten()

            # 4) Silence check
            amp = float(np.max(np.abs(audio)))
            if amp < 0.02:
                reply = "I couldn't hear anything. Could you try speaking a bit louder?"
                self.append_message(reply, is_user=False)
                threading.Thread(
                    target=self._speak_and_finish,
                    args=(reply,),
                    daemon=True
                ).start()
                return

            # 5) Show user bubble
            self.append_message("🎤 (audio captured)", is_user=True)

            # 6) Feature extraction & ONNX inference
            feats = extract_mfcc(array=audio, sr=sr)
            inp   = feats[np.newaxis, :, :]
            raw   = session.run(None, {input_name: inp})[0][0]
            exps  = np.exp(raw - np.max(raw))
            probs = exps / exps.sum()

            conf, idx = float(np.max(probs)), int(np.argmax(probs))
            emotion   = "Uncertain" if conf < 0.5 else CLASSES[idx]
            reply     = RESPONSES[emotion]

            print(f"[DEBUG] probs={probs}, conf={conf:.2f}, idx={idx}")
            self.append_message(reply, is_user=False)
            threading.Thread(
                target=self._speak_and_finish,
                args=(reply,),
                daemon=True
            ).start()

        except Exception as e:
            print("❌ record_and_predict crashed:", e)

        finally:
            # 7) Always reset button + icon + flag
            self._finish()
            # Note: _finish() does:
            #   self.button.setText("🎤 Record & Detect")
            #   self.button.setIcon(QIcon(self.mic_off))
            #   self.button.setEnabled(True)
            #   self.is_processing = False
            # So you don't need to duplicate that here.

    # def record_and_predict(self):
    #     if self.is_processing:
    #         return
    #     self.is_processing = True
    #     self.button.setEnabled(False)

    #     # 1) Turn mic icon ON
    #     self.mic_icon.setPixmap(self.mic_on)
    #     # self.button.setIcon(QIcon(self.mic_on))   # I can use this as well.
    #     self.button.setText(" 🔴 Recording...")    # **** new add for testing, no work then remove it
    #     QApplication.processEvents()

    #     try:
    #         # 2) Record 3s
    #         sr = FEATURE_SETTINGS['sample_rate']
    #         audio = sd.rec(int(3*sr), samplerate=sr, channels=1)
    #         sd.wait()
    #         audio = audio.flatten()
    #         amp = np.max(np.abs(audio))
    #         # 3) Silence check
    #         if  amp < 0.02:
    #             print(f"[DEBUG] silence branch (max amplitude={amp:.4f})")   # ← add this
    #             bot_reply = "I couldn't hear anything. Could you try speaking a bit louder?"

    #             print(f"[DEBUG] bot_reply (silence) = {bot_reply!r}") 
                
    #             self.append_message(bot_reply, is_user=False)

    #             # ✅ This will speak and then reset the state
    #             threading.Thread(
    #                 target=self._speak_and_finish,
    #                 args=(bot_reply,),
    #                 daemon=True
    #             ).start()
    #             return  # ✅ SAFE: _finish() will still be called after speech


    #         # 4) Show a user bubble placeholder
    #         self.append_message("(audio captured)", is_user=True)

    #         # 5) Feature extraction & inference
    #         feats = extract_mfcc_from_array(audio, sr)
    #         inp = feats[np.newaxis,:,:].astype(np.float32)
    #         if USE_ONNX:
    #             # probs = session.run(None, {input_name: inp})[0][0]     #*** no work then remove it 
    #             raw_logits = session.run(None, {input_name: inp})[0][0]
    #             exp_scores = np.exp(raw_logits - np.max(raw_logits))
    #             probs = exp_scores / exp_scores.sum()
    #             print("probs-", probs)
    #         else:
    #             import torch
    #             logits = model(torch.tensor(inp).unsqueeze(0))
    #             probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    #         conf= float(np.max(probs))
    #         pred_idx = int(np.argmax(probs))
    #         # DEBUG: print out to console
    #         print(f"[DEBUG] raw probs={probs}, conf={conf:.2f}, idx={pred_idx}")

    #         # 6) Map to emotion
    #         emotion = "Uncertain" if conf < 0.5 else CLASSES[pred_idx]
    #         bot_reply= RESPONSES[emotion]

    #         # 7) Show bot bubble
    #         self.append_message(bot_reply, is_user=False)

    #         # 8) Speak
    #         threading.Thread(
    #             target=self._speak_and_finish,
    #             args=(bot_reply,),
    #             daemon=True
    #         ).start()

    #     except Exception as e:
    #         self.label.setText("Error - Click 'Record' to retry", e)
    #         self.result_label.setText(f"Error: {str(e)}")
    #         print(f"Error: {e}")
    #         self._finish()
        
    #     finally:
    #         self.button.setText("Record & Detect")
    #         self.button.setIcon(QIcon(self.mic_off))
    #         self.button.setEnabled(True)
    #         self.is_processing = False
    #         self._finish()
            
    #     print(f"You said it with a {emotion.upper()} tone.")

# --- Run
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
