# import sys, os, json, threading, traceback, tempfile, time
# import numpy as np
# import librosa, sounddevice as sd, pyttsx3
# import soundfile as sf
# import requests
# import onnxruntime as ort
# import scipy.signal as signal
# from io import BytesIO
# from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
# from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QPushButton, QLabel,
#     QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, QSizePolicy
# )
# from extract_features import extract_mfcc
# from textblob import TextBlob

# # --- Project modules ---
# from config import FEATURE_SETTINGS, RESPONSES, PEPPER

# # ==========================================
# # CONFIGURATION
# # ==========================================
# DEBUG_AUDIO = False  # Set to False when done testing  #when testing is need then do the True set flag.

# try:
#     from config import ASSISTANT_STYLE
# except Exception:
#     ASSISTANT_STYLE = (
#         "You are Pepper, a warm, empathetic robot friend. "
#         "1. If the user is sad, validate them and ASK A QUESTION. "
#         "2. Keep replies under 2 sentences. "
#     )

# try:
#     from pepper_client import PepperClient
# except Exception:
#     PepperClient = None

# try:
#     TTS_ENGINE = pyttsx3.init()
# except Exception:
#     TTS_ENGINE = None

# # Runtime optimization
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
# USE_SENTIMENT_FUSION = True
# SENTIMENT_POS_THRESHOLD = 0.20   
# SENTIMENT_NEG_THRESHOLD = -0.20

# # Hallucination Filters
# BAD_PHRASES = [
#     "thank you", "thanks", "thank you for watching", "bye", 
#     "subtitle", "copyright", "amara", "community", "watching"
# ]

# # ---------------- Files & loading ----------------
# MODELS_DIR = "models"
# TRACKS = [
#     {"name": "SSL v1", "dir": os.path.join(MODELS_DIR, "ssl_v1"), "onnx": ["model_ssl.onnx"], "type": "ssl"},
#     {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], "type": "mfcc"},
# ]

# # ---------------- Utils ----------------
# def softmax(x):
#     x = x - np.max(x); e = np.exp(x); return e / e.sum()

# def _speak_async(text, on_done):
#     def run():
#         try:
#             if TTS_ENGINE:
#                 TTS_ENGINE.say(text)
#                 TTS_ENGINE.runAndWait()
#         except Exception:
#             traceback.print_exc()
#         finally:
#             QTimer.singleShot(0, on_done)
#     threading.Thread(target=run, daemon=True).start()

# def _read_json(path, default):
#     try:
#         with open(path, "r", encoding="utf-8") as f: return json.load(f)
#     except Exception: return default

# def _find_existing(paths):
#     for p in paths:
#         if os.path.exists(p): return p
#     return None

# def _bytes_to_audio(raw: bytes, sr_hint: int = 16000):
#     if len(raw) >= 12 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
#         y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
#         if y.ndim > 1: y = y[:, 0]
#         return y, sr
#     a = np.frombuffer(raw, dtype=np.int16)
#     if a.size == 0: raise ValueError("Pepper returned empty audio payload")
#     y = (a.astype(np.float32)) / 32768.0
#     return y, sr_hint

# # ==========================================
# # 2. THE SIGNAL CLEANER (High Pass Filter)
# # ==========================================
# def remove_fan_noise(y, sr):
#     """
#     Standard High-Pass Filter (120Hz).
#     Removes mechanical rumble but keeps voice.
#     """
#     try:
#         sos = signal.butter(10, 120, 'hp', fs=sr, output='sos')
#         cleaned = signal.sosfilt(sos, y)
#         return cleaned
#     except Exception:
#         return y

# # --- LOCAL ASR ---
# class LocalASR:
#     def __init__(self, model_size="base.en"):
#         from faster_whisper import WhisperModel
#         print(f"[ASR] Loading Whisper Model ({model_size})...")
#         self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
#     def transcribe(self, y, sr):
#         # NOTE: 'y' is assumed to be CLEANED by the worker before arriving here.
#         # This prevents double-filtering and allows us to save the clean audio to disk.
        
#         peak = float(np.max(np.abs(y)))
#         if peak < 0.01: return None 
        
#         # Normalize for Whisper
#         y_norm = (y / peak) * 0.95

#         fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
#         try:
#             sf.write(path, y_norm, sr, subtype="PCM_16")
#             segments, _ = self.model.transcribe(
#                 path, language="en", vad_filter=True, beam_size=5
#             )
#             full_text = " ".join([s.text for s in segments]).strip()
#             return full_text or None
#         except Exception:
#             return None
#         finally:
#             try: os.remove(path)
#             except Exception: pass

# class ChatEngine:
#     def __init__(self, model=None, host=None, debug=True):
#         self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
#         self.host  = (host  or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
#         self.url_gen  = f"{self.host}/api/generate"
#         self.system = ASSISTANT_STYLE
#         self.history = []
#         self.debug = debug

#     def _make_prompt(self, emotion: str, transcript: str | None) -> str:
#         t = (transcript or "").strip()
#         return (
#             f"{self.system}\n"
#             f"UserEmotion: {emotion or 'unknown'}\n"
#             f"UserSaid: {t if t else '(empty)'}\n"
#             "Respond naturally. If sad, ask a gentle question."
#         )

#     def _compose_text_prompt(self, prompt: str) -> str:
#         parts = [f"System: {self.system}"]
#         for m in self.history[-4:]: 
#             parts.append(f"{m['role'].capitalize()}: {m['content']}")
#         parts.append(f"User: {prompt}")
#         parts.append("Assistant:")
#         return "\n".join(parts)

#     def reply(self, emotion: str, transcript: str | None) -> str:
#         prompt = self._make_prompt(emotion, transcript)
#         try:
#             text_prompt = self._compose_text_prompt(prompt)
#             print(f"[Ollama] POST {self.url_gen}")
#             r = requests.post(
#                 self.url_gen,
#                 json={
#                     "model": self.model, "prompt": text_prompt, "stream": False,
#                     "options": {
#                         "temperature": 0.8, "num_predict": 50, "stop": ["\nUser:", "\nSystem:", "\nAssistant:"]
#                     },
#                 },
#                 timeout=60, 
#             )
#             r.raise_for_status()
#             text = (r.json().get("response") or "").strip()
#             if not text: raise RuntimeError("empty generate response")
#             self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
#             self.history = self.history[-6:] 
#             return text
#         except Exception:
#             return "I'm here. Could you say that again?"

# # ---------------- Chat UI Elements ----------------
# class ChatBubble(QLabel):
#     def __init__(self, text, is_user=False):
#         super().__init__(text)
#         self.setWordWrap(True)
#         self.setMaximumWidth(360)
#         self.setContentsMargins(10, 10, 10, 10)
#         font = QFont("Segoe UI", 10)
#         self.setFont(font)
#         if is_user:
#             self.setStyleSheet("background-color: #DCF8C6; color: black; border-radius: 12px; padding: 6px;")
#         else:
#             self.setStyleSheet("background-color: #EAEAEA; color: black; border-radius: 12px; padding: 6px;")

# class EmojiResult(QLabel):
#     def __init__(self, emotion):
#         super().__init__()
#         self.setAlignment(Qt.AlignCenter)
#         self.setStyleSheet("background-color: transparent;")
#         if emotion == "happy": self.setText("😃")
#         elif emotion == "sad": self.setText("😔")
#         else: self.setText("😐")
#         font = QFont("Segoe UI Emoji", 32) 
#         self.setFont(font)

# # ---------------- Main App ----------------
# class EmotionApp(QWidget):
#     sig_add_msg = pyqtSignal(str, bool)
#     sig_finish  = pyqtSignal()
#     sig_add_emoji = pyqtSignal(str)
#     sig_update_status = pyqtSignal(str)

#     def __init__(self):
#         super().__init__()
#         self._last_click = 0.0
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         icon_path = os.path.join(base_dir, "SER_Mental_Health_Robot.ico")   # or assets/app_icon.ico

#         if os.path.exists(icon_path):
#             self.setWindowIcon(QIcon(icon_path))
#         else:
#             print("[ICON] Not found:", icon_path)
#         self.setWindowTitle("Speech Emotion Application")
#         self.setGeometry(100, 100, 420, 600)
        
#         self.sig_add_msg.connect(self._add_msg)
#         self.sig_add_emoji.connect(self._add_emoji_bubble)
#         self.sig_update_status.connect(self.update_status_icon)
#         self.sig_finish.connect(self._finish_ui)

#         # Top Bar
    
#         top_bar = QHBoxLayout()
#         self.mic_icon = QLabel()
#         def load_icon(path, color):
#             if os.path.exists(path):
#                 return QPixmap(path).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#             else:
#                 pix = QPixmap(24, 24); pix.fill(QColor(color)); return pix
            
#         self.mic_off = load_icon("mic-off.png", "#808080")
#         self.mic_on  = load_icon("mic-on.png",  "#FF0000")
#         self.mic_icon.setPixmap(self.mic_off)

#         self.status_label = QLabel("😐")
#         self.status_label.setAlignment(Qt.AlignCenter)
#         self.status_label.setStyleSheet("font-size: 40px; color: #555555;")
#         top_bar.addStretch(); top_bar.addWidget(self.mic_icon); top_bar.addWidget(self.status_label); top_bar.addStretch()

#         # Chat area
#         self.chat_layout = QVBoxLayout()
#         self.chat_layout.setAlignment(Qt.AlignTop)
#         self.chat_layout.setSpacing(10)
#         container = QWidget(); container.setLayout(self.chat_layout); container.setStyleSheet("background-color: #FFFFFF;")
#         scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container); scroll.setStyleSheet("border: none;")
#         self.scroll = scroll

#         # Button
#         self.button = QPushButton("🎤 Record and Detect")
#         self.button.setMinimumHeight(50)
#         self.button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; font-size: 16px; border-radius: 8px; } QPushButton:disabled { background-color: #A0A0A0; }")
#         self.button.clicked.connect(self.record_and_predict)

#         main = QVBoxLayout(self); main.addLayout(top_bar); main.addWidget(scroll); main.addWidget(self.button)

#         self.is_processing = False
#         self.session = None
#         # Recording increased to 5 seconds
#         self.calib = {"mode":"threshold","sad_threshold":0.57,"min_confidence":0.50,"min_amp":0.1,"record_seconds":5.0}
#         self.temperature = 1.0
#         self.ssl = None

#         self.asr  = LocalASR()
#         self.chat = ChatEngine(debug=True)
#         self.dialog_phase = "opener"

#         self._auto_load_model()
#         self.pepper = None
#         if PEPPER.get("enabled") and PepperClient:
#             try:
#                 self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
#                 self.pepper.connect()
#             except Exception: self.pepper = None
    
#     def update_status_icon(self, state):
#         if state == "listening":
#             self.status_label.setText("🙂"); self.status_label.setStyleSheet("font-size: 40px; color: #3498db;")
#         elif state == "thinking":
#             self.status_label.setText("🤔"); self.status_label.setStyleSheet("font-size: 40px; color: #f1c40f;")
#         else:
#             self.status_label.setText("😐"); self.status_label.setStyleSheet("font-size: 40px; color: #888888;")

#     def _add_emoji_bubble(self, emotion):
#         emoji_widget = EmojiResult(emotion)
#         row = QHBoxLayout(); row.addStretch(); row.addWidget(emoji_widget); row.addStretch()
#         self.chat_layout.addLayout(row); self._scroll_to_bottom()
    
#     def _scroll_to_bottom(self):
#         QApplication.processEvents()
#         self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

#     def _say(self, text):
#         def finish_on_ui(): QTimer.singleShot(0, self._finish)
#         if self.pepper:
#             def run():
#                 try: self.pepper.tts(text)
#                 except Exception: traceback.print_exc()
#                 finally: finish_on_ui()
#             threading.Thread(target=run, daemon=True).start()
#         else:
#             _speak_async(text, finish_on_ui)

#     def on_prediction(self, label: str): pass
#     def _add_msg_safe(self, text, is_user=False): self.sig_add_msg.emit(text, is_user)
#     def _add_msg(self, text, is_user=False):
#         bubble = ChatBubble(text, is_user)
#         row = QHBoxLayout()
#         if is_user: row.addStretch(); row.addWidget(bubble)
#         else: row.addWidget(bubble); row.addStretch()
#         self.chat_layout.addLayout(row); self._scroll_to_bottom()

#     def _auto_load_model(self):
#         try:
#             chosen = None
#             for t in TRACKS:
#                 if onnx_path := _find_existing([os.path.join(t["dir"], fn) for fn in t["onnx"]]):
#                     chosen = (t, onnx_path); break
#             if not chosen: raise FileNotFoundError("No ONNX model found.")
#             track, onnx_path = chosen
#             self.model_type = track["type"]
#             print(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")

#             calib_path = os.path.join(track["dir"], "calibration.json")
#             self.calib.update(_read_json(calib_path, {}))
#             self.classes = _read_json(calib_path, {}).get("classes", ["happy", "sad"])
            
#             so = ort.SessionOptions()
#             so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.log_severity_level = 3
#             self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
#             self.input_name = self.session.get_inputs()[0].name
            
#             inp_shape = self.session.get_inputs()[0].shape
#             if self.model_type == "ssl" and inp_shape[-1] == 45: self.model_type = "mfcc"
#             if self.model_type == "mfcc" and inp_shape[-1] == 768: self.model_type = "ssl"

#             if self.model_type == "ssl" and self.ssl is None:
#                 import warnings; warnings.simplefilter("ignore")
#                 from ssl_frontend import SSLFrontend; self.ssl = SSLFrontend()
#         except Exception:
#             traceback.print_exc(); self._add_msg_safe("Setup problem.", is_user=False)

#     def _feat_mfcc(self, y, sr):
#         # NOTE: y is already cleaned in worker. Safe to use directly.
#         if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr): y = yt
#         except Exception: pass
#         return extract_mfcc(array=y, sr=sr)

#     def _feat_ssl(self, y, sr):
#         # NOTE: y is already cleaned in worker.
#         if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
#         return self.ssl(y)

#     def _finish_ui(self):
#         self.mic_icon.setPixmap(self.mic_off)
#         self.button.setText("🎤 Record and Detect")
#         self.button.setEnabled(True)
#         self.is_processing = False
#         self.sig_update_status.emit("idle")
#     def _finish(self): self.sig_finish.emit()

#     def record_and_predict(self):
#         now = time.monotonic()
#         if now - self._last_click < 0.8: return
#         self._last_click = now
#         if self.is_processing or self.session is None: return
#         self.is_processing = True
#         self.button.setEnabled(False)
#         self.button.setText("🔴  Listening…")
#         self.mic_icon.setPixmap(self.mic_on)
#         self.sig_update_status.emit("listening")
#         QApplication.processEvents()
#         threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

#     def _clean_transcript(self, s: str | None) -> str | None:
#         if not s: return None
#         t = s.strip(); t_low = t.lower()
#         if not t: return None
#         for bad in BAD_PHRASES:
#             if bad in t_low: return None
#         clean_text = t_low.replace(".", "").replace("!", "").replace("?", "").strip()
#         allowed_short = ["yes", "no", "hi", "hey", "ok", "sad", "bad", "mad", "joy", "cry", "wow", "fun"]
#         if len(clean_text) < 2: return None
#         if len(clean_text) < 5 and clean_text not in allowed_short: return None
#         return t

#     def _record_and_predict_worker(self):
#         try:
#             sr = FEATURE_SETTINGS.get("sample_rate", 16000)
#             dur = float(self.calib.get("record_seconds", 5.0)) 
#             use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
            
#             if use_pepper:
#                 try:
#                     raw = self.pepper.record(seconds=int(max(1, round(dur))), mode=PEPPER.get("record_mode", "seconds"))
#                     y, sr_file = _bytes_to_audio(raw, sr_hint=48000)
#                     if sr_file != sr: y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
#                 except Exception:
#                     y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()
#             else:
#                 y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()

#             # ==========================================
#             # 1. CLEAN AUDIO HERE (So we can save it)
#             # ==========================================
#             y = remove_fan_noise(y, sr)

#             # ==========================================
#             # 2. SAVE DEBUG AUDIO (Verify fan noise removal)
#             # ==========================================
#             if DEBUG_AUDIO:
#                 ts = int(time.time())
#                 fname = f"debug_audio_{ts}.wav"
#                 sf.write(fname, y, sr)
#                 print(f"[DEBUG] Saved processed audio to: {fname}")

#             peak = float(np.max(np.abs(y)))
#             if peak < 0.02:
#                 self._add_msg_safe("🎤 (too quiet)", is_user=True)
#                 self.sig_update_status.emit("idle")
#                 self._say("I couldn't hear you."); return

#             self.sig_update_status.emit("thinking")

#             # ASR
#             transcript = None
#             try:
#                 # We send the already-cleaned 'y' to transcribe
#                 raw_txt = self.asr.transcribe(y, sr)
#                 print(f"[ASR raw] {raw_txt!r}")
#                 transcript = self._clean_transcript(raw_txt)
#                 print(f"[ASR clean] {transcript!r}")
#             except Exception: transcript = None

#             if not transcript:
#                 msg = "I heard noise, but no words."
#                 self._add_msg_safe(msg, is_user=False)
#                 self.sig_update_status.emit("idle")
#                 self._say(msg); return

#             self._add_msg_safe(f"You: {transcript}", is_user=True)

#             # --- MODEL PREDICTION ---
#             feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
#             x = feats[np.newaxis, :, :].astype("float32")
#             logits = self.session.run(None, {self.input_name: x})[0][0]
#             T = max(1e-6, float(self.temperature))
#             probs = softmax(logits / T)
            
#             try: idx_h = self.classes.index("happy"); idx_s = self.classes.index("sad")
#             except ValueError: idx_h, idx_s = 0, 1
            
#             prob_happy = probs[idx_h]
#             prob_sad   = probs[idx_s]
#             print(f"[AI Model] Happy={prob_happy:.2f}, Sad={prob_sad:.2f}")

#             audio_label = "happy" if prob_happy > prob_sad else "sad"

#             # --- SENTIMENT FUSION ---
#             final_label = audio_label
#             if USE_SENTIMENT_FUSION:
#                 try:
#                     blob = TextBlob(transcript)
#                     polarity = blob.sentiment.polarity
#                     if polarity > SENTIMENT_POS_THRESHOLD: 
#                         final_label = "happy"
#                         print(f"[Sentiment] Overriding to HAPPY (polarity={polarity:.2f})")
#                     elif polarity < SENTIMENT_NEG_THRESHOLD: 
#                         final_label = "sad"
#                         print(f"[Sentiment] Overriding to SAD (polarity={polarity:.2f})")
#                 except Exception: pass

#             self.sig_add_emoji.emit(final_label)

#             if self.dialog_phase == "opener":
#                 reply = RESPONSES.get(final_label, "I am listening.")
#                 self.dialog_phase = "chat"
#             else:
#                 reply = self.chat.reply(final_label, transcript)

#             self._add_msg_safe(reply, is_user=False)
#             self._say(reply)
#             self.on_prediction(final_label)

#         except Exception:
#             traceback.print_exc(); self._add_msg_safe("Error processing audio.", is_user=False); self._finish()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     icon_path = os.path.join(base_dir, "SER_Mental_Health_Robot.ico")  # or .ico
#     if os.path.exists(icon_path):
#         app.setWindowIcon(QIcon(icon_path))
#     w = EmotionApp()
#     w.show()
#     sys.exit(app.exec_())

# gui_live_predict_frameless.py
# Integrated: Frameless UI (custom TitleBar) + your full SER + Pepper pipeline

import sys, os, json, threading, traceback, tempfile, time
import numpy as np
import librosa, sounddevice as sd, pyttsx3
import soundfile as sf
import requests
import onnxruntime as ort
import scipy.signal as signal
from io import BytesIO
from textblob import TextBlob

from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QToolButton, QStyle, QSizeGrip
)

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES, PEPPER
from extract_features import extract_mfcc

# ==========================================
# CONFIGURATION
# ==========================================
DEBUG_AUDIO = False

try:
    from config import ASSISTANT_STYLE
except Exception:
    ASSISTANT_STYLE = (
        "You are Pepper, a warm, empathetic robot friend. "
        "1. If the user is sad, validate them and ASK A QUESTION. "
        "2. Keep replies under 2 sentences. "
    )

try:
    from pepper_client import PepperClient
except Exception:
    PepperClient = None

try:
    TTS_ENGINE = pyttsx3.init()
except Exception:
    TTS_ENGINE = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

USE_SENTIMENT_FUSION = True
SENTIMENT_POS_THRESHOLD = 0.20
SENTIMENT_NEG_THRESHOLD = -0.20

BAD_PHRASES = [
    "thank you", "thanks", "thank you for watching", "bye",
    "subtitle", "copyright", "amara", "community", "watching"
]

MODELS_DIR = "models"
TRACKS = [
    {"name": "SSL v1", "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"], "type": "ssl"},
    {"name": "MFCC v1","dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx","model_mfcc_int8.onnx"], "type": "mfcc"},
]

# ==========================================
# Utils
# ==========================================
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def _speak_async(text, on_done):
    def run():
        try:
            if TTS_ENGINE:
                TTS_ENGINE.say(text)
                TTS_ENGINE.runAndWait()
        except Exception:
            traceback.print_exc()
        finally:
            QTimer.singleShot(0, on_done)
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

def _bytes_to_audio(raw: bytes, sr_hint: int = 16000):
    # WAV case
    if len(raw) >= 12 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
        y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        return y, int(sr)

    # Raw int16 PCM case
    a = np.frombuffer(raw, dtype=np.int16)
    if a.size == 0:
        raise ValueError("Pepper returned empty audio payload")
    y = (a.astype(np.float32)) / 32768.0
    return y, int(sr_hint)

def remove_fan_noise(y, sr):
    """High-pass (120 Hz) to remove rumble/fan noise."""
    try:
        sos = signal.butter(10, 120, 'hp', fs=sr, output='sos')
        return signal.sosfilt(sos, y)
    except Exception:
        return y

# ==========================================
# Local ASR
# ==========================================
class LocalASR:
    def __init__(self, model_size="base.en"):
        from faster_whisper import WhisperModel
        print(f"[ASR] Loading Whisper Model ({model_size})...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, y, sr):
        peak = float(np.max(np.abs(y)))
        if peak < 0.01:
            return None

        y_norm = (y / peak) * 0.95
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, y_norm, sr, subtype="PCM_16")
            segments, _ = self.model.transcribe(
                path, language="en", vad_filter=True, beam_size=5
            )
            full_text = " ".join([s.text for s in segments]).strip()
            return full_text or None
        except Exception:
            return None
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

# ==========================================
# Ollama chat engine
# ==========================================
class ChatEngine:
    def __init__(self, model=None, host=None, debug=True):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.url_gen = f"{self.host}/api/generate"
        self.system = ASSISTANT_STYLE
        self.history = []
        self.debug = debug

    def _make_prompt(self, emotion: str, transcript: str | None) -> str:
        t = (transcript or "").strip()
        return (
            f"{self.system}\n"
            f"UserEmotion: {emotion or 'unknown'}\n"
            f"UserSaid: {t if t else '(empty)'}\n"
            "Respond naturally. If sad, ask a gentle question."
        )

    def _compose_text_prompt(self, prompt: str) -> str:
        parts = [f"System: {self.system}"]
        for m in self.history[-4:]:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def reply(self, emotion: str, transcript: str | None) -> str:
        prompt = self._make_prompt(emotion, transcript)
        try:
            text_prompt = self._compose_text_prompt(prompt)
            if self.debug:
                print(f"[Ollama] POST {self.url_gen}")
            r = requests.post(
                self.url_gen,
                json={
                    "model": self.model,
                    "prompt": text_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 50,
                        "stop": ["\nUser:", "\nSystem:", "\nAssistant:"]
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            if not text:
                raise RuntimeError("empty generate response")

            self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
            self.history = self.history[-6:]
            return text
        except Exception:
            return "I'm here. Could you say that again?"

# ==========================================
# UI Widgets
# ==========================================
class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        self.setFont(QFont("Segoe UI", 10))
        if is_user:
            self.setStyleSheet("background:#DCF8C6; padding:10px; border-radius:12px;")
        else:
            self.setStyleSheet("background:#EAEAEA; padding:10px; border-radius:12px;")

class EmojiResult(QLabel):
    def __init__(self, emotion):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: transparent;")
        if emotion == "happy":
            self.setText("😃")
        elif emotion == "sad":
            self.setText("😔")
        else:
            self.setText("😐")
        self.setFont(QFont("Segoe UI Emoji", 32))

class TitleBar(QFrame):
    """Custom title bar: big icon + title, centered mic/status, window buttons, draggable."""
    def __init__(self, window, icon_path: str, title_text: str):
        super().__init__()
        self.window = window
        self._drag_pos = None
        self._mouse_pressed = False

        self.setObjectName("TitleBar")
        ICON = 110
        self.setFixedHeight(110)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 6)
        layout.setSpacing(10)

        # Big icon
        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(ICON, ICON)
        if os.path.exists(icon_path):
            self.icon_lbl.setPixmap(QPixmap(icon_path).scaled(
                ICON, ICON, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # Title
        self.title_lbl = QLabel(title_text)
        self.title_lbl.setFont(QFont("Poppins", 17, QFont.Bold))
        self.title_lbl.setObjectName("TitleText")

        # Center mic + status
        self.mic_lbl = QLabel("🎙️")
        self.mic_lbl.setFont(QFont("Segoe UI Emoji", 20))
        self.status_lbl = QLabel("😐")
        self.status_lbl.setFont(QFont("Segoe UI Emoji", 30))

        # ---- 3 zones ----
        left = QHBoxLayout()
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.icon_lbl)
        left.addWidget(self.title_lbl)
        left_widget = QWidget()
        left_widget.setLayout(left)

        center = QHBoxLayout()
        center.setSpacing(10)
        center.setContentsMargins(0, 0, 0, 0)
        center.addWidget(self.mic_lbl)
        center.addWidget(self.status_lbl)
        center_widget = QWidget()
        center_widget.setLayout(center)

        layout.addWidget(left_widget)
        layout.addStretch(1)
        layout.addWidget(center_widget, 0, Qt.AlignCenter)
        layout.addStretch(1)

        # Drag support
        self.icon_lbl.installEventFilter(self)
        self.title_lbl.installEventFilter(self)
        self.mic_lbl.installEventFilter(self)
        self.status_lbl.installEventFilter(self)

        layout.addSpacing(6)

        # Window buttons
        self.btn_min = QToolButton()
        self.btn_max = QToolButton()
        self.btn_close = QToolButton()
        st = self.style()
        self.btn_min.setIcon(st.standardIcon(QStyle.SP_TitleBarMinButton))
        self.btn_max.setIcon(st.standardIcon(QStyle.SP_TitleBarMaxButton))
        self.btn_close.setIcon(st.standardIcon(QStyle.SP_TitleBarCloseButton))

        self.btn_min.clicked.connect(self.window.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max_restore)
        self.btn_close.clicked.connect(self.window.close)

        for b in (self.btn_min, self.btn_max, self.btn_close):
            b.setFixedSize(34, 28)
            b.setAutoRaise(True)

        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

        self.setStyleSheet("""
        QFrame#TitleBar {
            background: #f4f6f8;
            border-bottom: 1px solid #e3e7ee;
        }
        QLabel#TitleText { color: #1f2d3d; }
        QToolButton { border-radius: 6px; }
        QToolButton:hover { background: #e9eef6; }
        QToolButton:pressed { background: #dbe6f6; }
        """)

        self._sync_max_button()

    def set_state(self, state: str):
        if state == "listening":
            self.mic_lbl.setText("🔴")
            self.status_lbl.setText("🙂")
        elif state == "thinking":
            self.mic_lbl.setText("🎙️")
            self.status_lbl.setText("🤔")
        else:
            self.mic_lbl.setText("🎙️")
            self.status_lbl.setText("😐")

    def _toggle_max_restore(self):
        if self.window.windowState() & Qt.WindowMaximized:
            self.window.showNormal()
        else:
            self.window.showMaximized()
        self._sync_max_button()

    def _sync_max_button(self):
        if self.window.windowState() & Qt.WindowMaximized:
            self.btn_max.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        else:
            self.btn_max.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._mouse_pressed = True
            self._drag_pos = e.globalPos() - self.window.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._mouse_pressed and self._drag_pos is not None and not self.window.isMaximized():
            self.window.move(e.globalPos() - self._drag_pos)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._mouse_pressed = False
        self._drag_pos = None
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._toggle_max_restore()
        super().mouseDoubleClickEvent(e)

    def eventFilter(self, obj, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self._mouse_pressed = True
            self._drag_pos = event.globalPos() - self.window.frameGeometry().topLeft()
            return True

        if event.type() == event.MouseMove and self._mouse_pressed and self._drag_pos is not None:
            if not self.window.isMaximized():
                self.window.move(event.globalPos() - self._drag_pos)
            return True

        if event.type() == event.MouseButtonRelease:
            self._mouse_pressed = False
            self._drag_pos = None
            return True

        return super().eventFilter(obj, event)

# ==========================================
# Main App (Frameless UI + Pipeline)
# ==========================================
class EmotionAppFrameless(QWidget):
    sig_add_msg = pyqtSignal(str, bool)
    sig_finish = pyqtSignal()
    sig_add_emoji = pyqtSignal(str)
    sig_update_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._last_click = 0.0
        self.is_processing = False
        self.session = None
        self.ssl = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        header_icon_path = os.path.join(base_dir, "robot_logo.png")
        app_icon_ico = os.path.join(base_dir, "SER_Mental_Health_Robot.ico")

        if os.path.exists(app_icon_ico):
            self.setWindowIcon(QIcon(app_icon_ico))

        self.setWindowTitle("Speech Emotion Application")

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setMinimumSize(860, 820)

        self.sig_add_msg.connect(self._add_msg)
        self.sig_add_emoji.connect(self._add_emoji_bubble)
        self.sig_update_status.connect(self.update_status_icon)
        self.sig_finish.connect(self._finish_ui)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        container = QFrame()
        container.setObjectName("Container")
        container.setStyleSheet("""
        QFrame#Container {
            background: white;
            border: 1px solid #dfe6ee;
            border-radius: 15px;
        }
        """)
        outer.addWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.titlebar = TitleBar(self, header_icon_path, "Speech Emotion Application")
        layout.addWidget(self.titlebar)

        content = QFrame()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)

        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)

        chat_container = QWidget()
        chat_container.setLayout(self.chat_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(chat_container)
        scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #e4e7ec; border-radius: 10px; background: #ffffff; }
        """)
        self.scroll = scroll

        self.button = QPushButton("🎤  Record and Detect")
        self.button.setMinimumHeight(60)
        self.button.setStyleSheet("""
            QPushButton { background:#0078D7; color:white; font-size:16px; border-radius:10px; }
            QPushButton:disabled { background:#A0A0A0; }
            QPushButton:pressed { background:#0b5aa0; }
        """)
        self.button.clicked.connect(self.record_and_predict)

        content_layout.addWidget(scroll, 1)
        content_layout.addWidget(self.button)

        layout.addWidget(content, 1)

        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(18, 18)
        self.size_grip.raise_()

        self.calib = {
            "mode": "threshold",
            "sad_threshold": 0.57,
            "min_confidence": 0.50,
            "min_amp": 0.1,
            "record_seconds": 5.0
        }
        self.temperature = 1.0
        self.dialog_phase = "opener"

        self.asr = LocalASR()
        self.chat = ChatEngine(debug=True)

        self._auto_load_model()

        self.pepper = None
        if PEPPER.get("enabled") and PepperClient:
            try:
                self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
                self.pepper.connect()
            except Exception:
                self.pepper = None

        self._add_msg_safe("Welcome to Social Companion Robot. Press Record and Detect to start.", is_user=False)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        margin = 10
        self.size_grip.move(
            self.width() - self.size_grip.width() - margin,
            self.height() - self.size_grip.height() - margin
        )

    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            self.titlebar._sync_max_button()
        super().changeEvent(e)

    def update_status_icon(self, state):
        self.titlebar.set_state(state)

    def _add_emoji_bubble(self, emotion):
        emoji_widget = EmojiResult(emotion)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(emoji_widget)
        row.addStretch()
        self.chat_layout.addLayout(row)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QApplication.processEvents()
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def _add_msg_safe(self, text, is_user=False):
        self.sig_add_msg.emit(text, is_user)

    def _add_msg(self, text, is_user=False):
        bubble = ChatBubble(text, is_user)
        row = QHBoxLayout()
        if is_user:
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()
        self.chat_layout.addLayout(row)
        self._scroll_to_bottom()

    def _finish_ui(self):
        self.button.setText("🎤  Record and Detect")
        self.button.setEnabled(True)
        self.is_processing = False
        self.sig_update_status.emit("idle")

    def _finish(self):
        self.sig_finish.emit()

    def _say(self, text):
        def finish_on_ui():
            QTimer.singleShot(0, self._finish)

        if self.pepper:
            def run():
                try:
                    self.pepper.tts(text)
                except Exception:
                    traceback.print_exc()
                finally:
                    finish_on_ui()
            threading.Thread(target=run, daemon=True).start()
        else:
            _speak_async(text, finish_on_ui)

    def _auto_load_model(self):
        try:
            chosen = None
            for t in TRACKS:
                onnx_path = _find_existing([os.path.join(t["dir"], fn) for fn in t["onnx"]])
                if onnx_path:
                    chosen = (t, onnx_path)
                    break

            if not chosen:
                raise FileNotFoundError("No ONNX model found in TRACKS folders.")

            track, onnx_path = chosen
            self.model_type = track["type"]
            print(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")

            calib_path = os.path.join(track["dir"], "calibration.json")
            self.calib.update(_read_json(calib_path, {}))
            self.classes = _read_json(calib_path, {}).get("classes", ["happy", "sad"])

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name

            inp_shape = self.session.get_inputs()[0].shape
            if self.model_type == "ssl" and inp_shape[-1] == 45:
                self.model_type = "mfcc"
            if self.model_type == "mfcc" and inp_shape[-1] == 768:
                self.model_type = "ssl"

            if self.model_type == "ssl" and self.ssl is None:
                import warnings
                warnings.simplefilter("ignore")
                from ssl_frontend import SSLFrontend
                self.ssl = SSLFrontend()

        except Exception:
            traceback.print_exc()
            self._add_msg_safe("Setup problem: model could not be loaded.", is_user=False)

    def _feat_mfcc(self, y, sr):
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        try:
            yt, _ = librosa.effects.trim(y, top_db=30)
            if len(yt) > int(0.25 * sr):
                y = yt
        except Exception:
            pass
        return extract_mfcc(array=y, sr=sr)

    def _feat_ssl(self, y, sr):
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        return self.ssl(y)

    def record_and_predict(self):
        now = time.monotonic()
        if now - self._last_click < 0.8:
            return
        self._last_click = now

        if self.is_processing or self.session is None:
            return

        self.is_processing = True
        self.button.setEnabled(False)
        self.button.setText("🔴  Listening…")
        self.sig_update_status.emit("listening")
        QApplication.processEvents()

        threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

    def _clean_transcript(self, s: str | None) -> str | None:
        if not s:
            return None
        t = s.strip()
        t_low = t.lower()
        if not t:
            return None
        for bad in BAD_PHRASES:
            if bad in t_low:
                return None
        clean_text = t_low.replace(".", "").replace("!", "").replace("?", "").strip()
        allowed_short = ["yes", "no", "hi", "hey", "ok", "sad", "bad", "mad", "joy", "cry", "wow", "fun"]
        if len(clean_text) < 2:
            return None
        if len(clean_text) < 5 and clean_text not in allowed_short:
            return None
        return t

    def _record_and_predict_worker(self):
        try:
            sr = int(FEATURE_SETTINGS.get("sample_rate", 16000))      # pipeline SR (your models)
            dur = float(self.calib.get("record_seconds", 5.0))
            use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))

            # ---------------- Record ----------------
            if use_pepper:
                try:
                    raw = self.pepper.record(
                        seconds=int(max(1, round(dur))),
                        mode=PEPPER.get("record_mode", "seconds")
                    )

                    # ✅ FIX: use config sample_rate as sr_hint, NOT hardcoded 48000
                    pepper_sr_hint = int(PEPPER.get("sample_rate", sr))
                    y, sr_file = _bytes_to_audio(raw, sr_hint=pepper_sr_hint)

                    # resample to pipeline SR
                    if sr_file != sr:
                        y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)

                except Exception:
                    y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
                    sd.wait()
                    y = y.flatten()
            else:
                y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                y = y.flatten()

            # ---------------- Clean (only once, using pipeline SR) ----------------
            y = remove_fan_noise(y, sr)

            # Debug save
            if DEBUG_AUDIO:
                ts = int(time.time())
                fname = f"debug_audio_{ts}.wav"
                sf.write(fname, y, sr)
                print(f"[DEBUG] Saved processed audio to: {fname}")

            peak = float(np.max(np.abs(y)))
            if peak < 0.02:
                self._add_msg_safe("🎤 (too quiet)", is_user=True)
                self.sig_update_status.emit("idle")
                self._say("I couldn't hear you.")
                return

            self.sig_update_status.emit("thinking")

            # ---------------- ASR ----------------
            transcript = None
            try:
                raw_txt = self.asr.transcribe(y, sr)
                print(f"[ASR raw] {raw_txt!r}")
                transcript = self._clean_transcript(raw_txt)
                print(f"[ASR clean] {transcript!r}")
            except Exception:
                transcript = None

            if not transcript:
                msg = "I heard noise, but no words."
                self._add_msg_safe(msg, is_user=False)
                self.sig_update_status.emit("idle")
                self._say(msg)
                return

            self._add_msg_safe(f"You: {transcript}", is_user=True)

            # ---------------- Predict ----------------
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")

            logits = self.session.run(None, {self.input_name: x})[0][0]
            T = max(1e-6, float(self.temperature))
            probs = softmax(logits / T)

            try:
                idx_h = self.classes.index("happy")
                idx_s = self.classes.index("sad")
            except ValueError:
                idx_h, idx_s = 0, 1

            prob_happy = float(probs[idx_h])
            prob_sad = float(probs[idx_s])
            print(f"[AI Model] Happy={prob_happy:.2f}, Sad={prob_sad:.2f}")

            audio_label = "happy" if prob_happy > prob_sad else "sad"

            # ---------------- Sentiment fusion ----------------
            final_label = audio_label
            if USE_SENTIMENT_FUSION:
                try:
                    polarity = TextBlob(transcript).sentiment.polarity
                    if polarity > SENTIMENT_POS_THRESHOLD:
                        final_label = "happy"
                        print(f"[Sentiment] Override HAPPY (polarity={polarity:.2f})")
                    elif polarity < SENTIMENT_NEG_THRESHOLD:
                        final_label = "sad"
                        print(f"[Sentiment] Override SAD (polarity={polarity:.2f})")
                except Exception:
                    pass

            self.sig_add_emoji.emit(final_label)

            # ---------------- Reply ----------------
            if self.dialog_phase == "opener":
                reply = RESPONSES.get(final_label, "I am listening.")
                self.dialog_phase = "chat"
            else:
                reply = self.chat.reply(final_label, transcript)

            self._add_msg_safe(reply, is_user=False)
            self._say(reply)

        except Exception:
            traceback.print_exc()
            self._add_msg_safe("Error processing audio.", is_user=False)
            self._finish()

# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_icon_ico = os.path.join(base_dir, "SER_Mental_Health_Robot.ico")
    if os.path.exists(app_icon_ico):
        app.setWindowIcon(QIcon(app_icon_ico))

    w = EmotionAppFrameless()
    w.show()
    sys.exit(app.exec_())
