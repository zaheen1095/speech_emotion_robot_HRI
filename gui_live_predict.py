# gui_live_predict.py
import sys, os, json, threading, traceback
import numpy as np
import librosa, sounddevice as sd, pyttsx3

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon,QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
)

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES
from extract_features import extract_mfcc
# from ssl_frontend import SSLFrontend  # used only if SSL model found

import onnxruntime as ort

# ---------------- Files & loading ----------------
MODELS_DIR = "models"
TRACKS = [
    {"name": "SSL v1",  "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"],           "type": "ssl"},
    {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], "type": "mfcc"},
]

def softmax(x):
    x = x - np.max(x); e = np.exp(x); return e / e.sum()

def _speak_async(text, on_done):
    def run():
        try:
            eng = pyttsx3.init()
            eng.say(text); eng.runAndWait()
        except Exception:
            traceback.print_exc()
        finally:
            on_done()
    threading.Thread(target=run, daemon=True).start()

def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ---------------- Chat UI ----------------
class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        color = "#e0e0e0" if is_user else "#95abbe"
        self.setStyleSheet(f"background:{color}; border-radius:10px; padding:8px;")

# ---------------- Main App ----------------
class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
    
        self.setWindowTitle("Speech Emotion Detection Application")
        self.setGeometry(100, 100, 400, 500)

        # top bar
        self.mic_icon = QLabel()
        self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mic_icon.setPixmap(self.mic_off)

        # Chat area
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        container = QWidget(); container.setLayout(self.chat_layout)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container)
        self.scroll = scroll

        # Record button (single control)
        self.button = QPushButton("🎤  Record & Detect")
        self.button.setIcon(QIcon("mic-off.png"))
        self.button.setIconSize(QSize(20, 20))
        self.button.clicked.connect(self.record_and_predict)

        # Layout
        # head = QHBoxLayout(); head.addWidget(title); head.addStretch()
        main = QVBoxLayout(self)
        main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
        # main.addLayout(head)
        main.addWidget(scroll)
        main.addWidget(self.button)

        # Runtime state
        self.is_processing = False
        self.session = None
        self.input_name = None
        self.model_type = None       # "ssl" or "mfcc"
        self.classes = ["happy", "sad"]
        self.calib = {"mode":"threshold","sad_threshold":0.57,"min_confidence":0.50,"min_amp":0.02,"record_seconds":3.0}
        self.temperature = 1.0
        self.ssl = None              # SSLFrontend (lazy)

        self._auto_load_model()

    # ---- Robot integration hook (use later) ----
    def on_prediction(self, label: str):
        """
        Placeholder for robot integration.
        E.g., send event to behavior planner:
        robot.handle_emotion(label)
        """
        pass

    # ---- Minimal chat helpers ----
    def _add_msg(self, text, is_user=False):
        bubble = ChatBubble(text, is_user)
        row = QHBoxLayout()
        if is_user:
            row.addStretch(); row.addWidget(bubble)
        else:
            row.addWidget(bubble); row.addStretch()
        f = QFrame(); f.setLayout(row)
        self.chat_layout.addWidget(f)
        sb = self.scroll.verticalScrollBar(); sb.setValue(sb.maximum())

    # ---- Model load (auto, silent) ----
    def _auto_load_model(self):
        try:
            # Prefer SSL, fallback to MFCC
            chosen = None
            for t in TRACKS:
                tdir = t["dir"]
                onnx_path = _find_existing([os.path.join(tdir, fn) for fn in t["onnx"]])
                if onnx_path:
                    chosen = (t, onnx_path); break
            if not chosen:
                raise FileNotFoundError("No ONNX model found in models/ssl_v1 or models/mfcc_v1.")

            track, onnx_path = chosen
            self.model_type = track["type"]

            print(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")


            # load per-track calibration + optional temperature
            calib_path = os.path.join(track["dir"], "calibration.json")
            self.calib.update(_read_json(calib_path, {}))
            temp = _read_json(os.path.join(track["dir"], "temperature.json"), {"temperature":1.0})
            self.temperature = float(temp.get("temperature", 1.0))

            # classes order (optional in calibration)
            self.classes = _read_json(calib_path, {}).get("classes", self.classes)

            # ORT session
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.log_severity_level = 3
            self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name

            # lazy SSL frontend if needed
            if self.model_type == "ssl" and self.ssl is None:
                # reduce HF noise
                import warnings
                try:
                    from transformers.utils import logging as hf_logging
                    hf_logging.set_verbosity_error()
                except Exception:
                    pass
                warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing`.*")
                warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces`.*")
                from ssl_frontend import SSLFrontend
                self.ssl = SSLFrontend()  # let your class pick model & device

        except Exception as e:
            traceback.print_exc()
            self._add_msg("Setup problem. Please check model files.", is_user=False)

    # ---- Features (private) ----
    def _feat_mfcc(self, y, sr):
        # ensure target sr
        target = FEATURE_SETTINGS.get("sample_rate", 16000)
        if sr != target:
            y = librosa.resample(y, orig_sr=sr, target_sr=target); sr = target
        # conservative trim
        try:
            yt, _ = librosa.effects.trim(y, top_db=30)
            if len(yt) > int(0.25 * sr): y = yt
        except: pass
        return extract_mfcc(array=y, sr=sr)  # (T,D)

    def _feat_ssl(self, y, sr):
        # SSLFrontend typically expects 16 kHz mono waveform
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
        try:
            yt, _ = librosa.effects.trim(y, top_db=30)
            if len(yt) > int(0.25 * sr): y = yt
        except: pass
        return self.ssl(y)  # (T,D) embeddings

    # ---- Decoding (silent; no probs shown) ----
    def _decode(self, probs):
        # class order from calibration if provided
        try:
            idx_h = self.classes.index("happy"); idx_s = self.classes.index("sad")
        except ValueError:
            idx_h, idx_s = 0, 1
        p_h, p_s = float(probs[idx_h]), float(probs[idx_s])
        p_max = max(p_h, p_s)
        if p_max < float(self.calib.get("min_confidence", 0.50)):
            return "Uncertain"
        if self.calib.get("mode", "threshold") == "threshold":
            return "sad" if p_s >= float(self.calib.get("sad_threshold", 0.57)) else "happy"
        return "happy" if p_h >= p_s else "sad"

    def _finish(self):
        self.mic_icon.setPixmap(self.mic_off)
        self.button.setText("🎤  Record & Detect")
        self.button.setIcon(QIcon("mic-off.png"))
        self.button.setEnabled(True)
        self.is_processing = False

    # ---- Main action ----
    def record_and_predict(self):
        if self.is_processing or self.session is None:
            return
        self.is_processing = True
        self.button.setEnabled(False)
        self.button.setText("🔴  Listening…")
        self.button.setIcon(QIcon("mic-on.png"))
        self.mic_icon.setPixmap(self.mic_on)
        QApplication.processEvents()

        try:
            # record
            sr = FEATURE_SETTINGS.get("sample_rate", 16000)
            dur = float(self.calib.get("record_seconds", 3.0))
            y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            y = y.flatten()

            # audibility gate
            if float(np.max(np.abs(y))) < float(self.calib.get("min_amp", 0.02)):
                reply = RESPONSES.get("Uncertain",
                        "I couldn't hear clearly. Please speak a bit closer to the mic.")
                self._add_msg(reply, is_user=False)
                _speak_async(reply, self._finish)
                return

            self._add_msg("🎤 (audio captured)", is_user=True)
            QApplication.processEvents()
            # features based on loaded model type
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")  # [1,T,D]

            # inference → temperature → probs → decode
            logits = self.session.run(None, {self.input_name: x})[0][0]
            T = max(1e-6, float(self.temperature))
            probs = softmax(logits / T)
            label = self._decode(probs)

            # text reply (no numbers)
            reply = RESPONSES.get(label, "I'm here with you.")
            self._add_msg(reply, is_user=False)
            self.on_prediction(label)  # hook for robot behavior

            _speak_async(reply, self._finish)

        except Exception:
            traceback.print_exc()
            self._add_msg("Something went wrong. Let's try again.", is_user=False)
            self._finish()

# ---- Run ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = EmotionApp()
    w.show()
    sys.exit(app.exec_())
