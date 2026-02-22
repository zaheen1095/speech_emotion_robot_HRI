import sys, os, json, threading, traceback, tempfile, time, logging
from io import BytesIO

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import requests
import onnxruntime as ort
import scipy.signal as signal
from textblob import TextBlob
import pyttsx3

from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QToolButton, QStyle, QSizeGrip
)

# =========================
# Project modules
# =========================
try:
    from config import FEATURE_SETTINGS, RESPONSES, PEPPER, ASSISTANT_STYLE
except Exception:
    FEATURE_SETTINGS = {"sample_rate": 16000}
    RESPONSES = {"happy": "I’m glad you sound happy.", "sad": "I’m sorry you’re feeling this way. What happened?"}
    PEPPER = {"enabled": False, "use_pepper_mic": False, "ip": "127.0.0.1", "port": 9559, "record_mode": "seconds", "sample_rate": 48000}
    ASSISTANT_STYLE = (
        "You are Pepper, a warm, empathetic robot friend. "
        "If the user is sad, validate them and ask one gentle question. "
        "Keep replies under 2 sentences."
    )

try:
    from extract_features import extract_mfcc
except Exception:
    # placeholder if not available; won't be used if your project module exists
    def extract_mfcc(array, sr):
        return np.zeros((313, 45), dtype=np.float32)

try:
    from pepper_client import PepperClient
except Exception:
    PepperClient = None

# SSL frontend loaded lazily only if SSL model is chosen
SSLFrontend = None

# =========================
# Logging (terminal + optional file)
# =========================
LOG_TO_FILE = True
LOG_FILE = os.path.join(os.getcwd(), "ser_terminal.log")

logger = logging.getLogger("SER")
logger.setLevel(logging.INFO)
handler_console = logging.StreamHandler(sys.stdout)
handler_console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(handler_console)

if LOG_TO_FILE:
    handler_file = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    handler_file.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler_file)
    logger.info(f"[LOG] Writing terminal log to: {LOG_FILE}")

# =========================
# Config knobs
# =========================
DEBUG_AUDIO_SAVE = False
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

# =========================
# Utils
# =========================
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def remove_fan_noise(y, sr):
    try:
        sos = signal.butter(10, 120, 'hp', fs=sr, output='sos')
        return signal.sosfilt(sos, y)
    except Exception:
        return y

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
    # WAV
    if len(raw) >= 12 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
        y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        return y, int(sr)

    # raw PCM int16
    a = np.frombuffer(raw, dtype=np.int16)
    if a.size == 0:
        raise ValueError("Pepper returned empty audio payload")
    y = (a.astype(np.float32)) / 32768.0
    return y, int(sr_hint)

def _finite_audio(y):
    if y is None or y.size == 0:
        return False
    return np.isfinite(y).all()

def list_input_devices_once():
    try:
        devices = sd.query_devices()
        logger.info("[AUDIO] ---- Available INPUT devices ----")
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                sr = int(d.get("default_samplerate", 0))
                name = d.get("name", "")
                logger.info(f"[AUDIO] {i:>2} | in={d['max_input_channels']} | sr={sr} | {name}")
        logger.info(f"[AUDIO] Default input device: {sd.default.device}")
        logger.info("[AUDIO] --------------------------------")
    except Exception as e:
        logger.info(f"[AUDIO] device query failed: {e}")

def record_laptop(duration_s: float, target_sr: int):
    """
    Robust laptop recording:
    - uses explicit input device if SER_INPUT_DEVICE is set
    - otherwise uses sd.default.device[0] if valid
    - records at device default SR, then resamples to target_sr
    """
    env_dev = os.getenv("SER_INPUT_DEVICE", "").strip()
    device = None
    if env_dev.isdigit():
        device = int(env_dev)

    # pick device default sr
    if device is not None:
        info = sd.query_devices(device, "input")
        native_sr = int(info["default_samplerate"])
    else:
        # try default input device index
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        if isinstance(default_in, (list, tuple)):
            default_in = default_in[0]
        try:
            info = sd.query_devices(default_in, "input")
            device = default_in
            native_sr = int(info["default_samplerate"])
        except Exception:
            # fallback: first input device
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    device = i
                    native_sr = int(d.get("default_samplerate", 44100))
                    break
            if device is None:
                raise RuntimeError("No input microphone device found.")

    logger.info(f"[REC] laptop device={device} native_sr={native_sr} -> target_sr={target_sr}")

    y = sd.rec(int(duration_s * native_sr), samplerate=native_sr, channels=1, dtype="float32", device=device)
    sd.wait()
    y = y.reshape(-1)

    if not _finite_audio(y):
        raise RuntimeError("Audio buffer is not finite everywhere (NaN/Inf).")

    if native_sr != target_sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=target_sr)

    return y.astype(np.float32), target_sr

# =========================
# Local ASR (Whisper)
# =========================
class LocalASR:
    def __init__(self, model_size="base.en"):
        from faster_whisper import WhisperModel
        logger.info(f"[ASR] Loading Whisper Model ({model_size})...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, y, sr):
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak < 0.01:
            return None

        y_norm = (y / peak) * 0.95
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, y_norm, sr, subtype="PCM_16")
            segments, _ = self.model.transcribe(path, language="en", vad_filter=True, beam_size=5)
            full_text = " ".join([s.text for s in segments]).strip()
            return full_text or None
        except Exception:
            return None
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

# =========================
# Ollama Chat Engine
# =========================
class ChatEngine:
    def __init__(self, model=None, host=None, debug=True):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.url_gen = f"{self.host}/api/generate"
        self.system = ASSISTANT_STYLE
        self.history = []
        self.debug = debug

    def _make_prompt(self, emotion, transcript):
        t = (transcript or "").strip()
        return (
            f"{self.system}\n"
            f"UserEmotion: {emotion or 'unknown'}\n"
            f"UserSaid: {t if t else '(empty)'}\n"
            "Respond naturally. If sad, ask a gentle question."
        )

    def _compose_text_prompt(self, prompt):
        parts = [f"System: {self.system}"]
        for m in self.history[-4:]:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def reply(self, emotion, transcript):
        prompt = self._make_prompt(emotion, transcript)
        try:
            text_prompt = self._compose_text_prompt(prompt)
            if self.debug:
                logger.info(f"[Ollama] POST {self.url_gen}")
            r = requests.post(
                self.url_gen,
                json={
                    "model": self.model,
                    "prompt": text_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 60,
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
            return "I’m here with you. Could you say that again?"

# =========================
# UI Widgets
# =========================
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

        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(ICON, ICON)
        if os.path.exists(icon_path):
            self.icon_lbl.setPixmap(QPixmap(icon_path).scaled(ICON, ICON, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.title_lbl = QLabel(title_text)
        self.title_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.title_lbl.setObjectName("TitleText")

        self.mic_lbl = QLabel("🎙️")
        self.mic_lbl.setFont(QFont("Segoe UI Emoji", 20))
        self.status_lbl = QLabel("😐")
        self.status_lbl.setFont(QFont("Segoe UI Emoji", 30))

        left = QHBoxLayout()
        left.addWidget(self.icon_lbl)
        left.addWidget(self.title_lbl)
        left_w = QWidget(); left_w.setLayout(left)

        center = QHBoxLayout()
        center.addWidget(self.mic_lbl)
        center.addWidget(self.status_lbl)
        center_w = QWidget(); center_w.setLayout(center)

        layout.addWidget(left_w)
        layout.addStretch(1)
        layout.addWidget(center_w, 0, Qt.AlignCenter)
        layout.addStretch(1)

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

        self.icon_lbl.installEventFilter(self)
        self.title_lbl.installEventFilter(self)
        self.mic_lbl.installEventFilter(self)
        self.status_lbl.installEventFilter(self)

        self.setStyleSheet("""
        QFrame#TitleBar { background: #f4f6f8; border-bottom: 1px solid #e3e7ee; }
        QLabel#TitleText { color: #1f2d3d; }
        QToolButton { border-radius: 6px; }
        QToolButton:hover { background: #e9eef6; }
        QToolButton:pressed { background: #dbe6f6; }
        """)

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

# =========================
# Main App
# =========================
class EmotionAppFrameless(QWidget):
    sig_add_msg = pyqtSignal(str, bool)
    sig_add_emoji = pyqtSignal(str)
    sig_state = pyqtSignal(str)
    sig_finish = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._last_click = 0.0
        self.is_processing = False

        # Window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setMinimumSize(860, 820)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        header_icon_path = os.path.join(base_dir, "robot_logo.png")
        app_icon_ico = os.path.join(base_dir, "SER_Mental_Health_Robot.ico")
        if os.path.exists(app_icon_ico):
            self.setWindowIcon(QIcon(app_icon_ico))

        # Signals
        self.sig_add_msg.connect(self._add_msg)
        self.sig_add_emoji.connect(self._add_emoji_bubble)
        self.sig_state.connect(self._set_state)
        self.sig_finish.connect(self._finish_ui)

        # Layout shell
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        container = QFrame()
        container.setObjectName("Container")
        container.setStyleSheet("QFrame#Container { background: white; border: 1px solid #dfe6ee; border-radius: 15px; }")
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

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(chat_container)
        self.scroll.setStyleSheet("QScrollArea { border: 1px solid #e4e7ec; border-radius: 10px; background: #ffffff; }")

        self.button = QPushButton("🎤  Record and Detect")
        self.button.setMinimumHeight(60)
        self.button.setStyleSheet("""
            QPushButton { background:#0078D7; color:white; font-size:16px; border-radius:10px; }
            QPushButton:disabled { background:#A0A0A0; }
            QPushButton:pressed { background:#0b5aa0; }
        """)
        self.button.clicked.connect(self.record_and_predict)

        content_layout.addWidget(self.scroll, 1)
        content_layout.addWidget(self.button)
        layout.addWidget(content, 1)

        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(18, 18)
        self.size_grip.raise_()

        # Pipeline config
        self.calib = {"record_seconds": 5.0}
        self.temperature = 1.0
        self.dialog_phase = "opener"

        # Backend objects
        # list_input_devices_once()
        self.asr = LocalASR()
        self.chat = ChatEngine(debug=True)

        self.session = None
        self.input_name = None
        self.model_type = None
        self.classes = ["happy", "sad"]
        self.ssl = None

        self._auto_load_model()

        # Pepper
        self.pepper = None
        if PEPPER.get("enabled") and PepperClient:
            try:
                self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
                self.pepper.connect()
                logger.info("[PEPPER] Connected.")
            except Exception as e:
                self.pepper = None
                logger.info(f"[PEPPER] Not available; using laptop mic/TTS. ({e})")
        else:
            logger.info("[PEPPER] Not available; using laptop mic/TTS.")

        self.sig_add_msg.emit("Welcome. Press Record and Detect to start.", False)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        margin = 10
        self.size_grip.move(self.width() - self.size_grip.width() - margin, self.height() - self.size_grip.height() - margin)

    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            pass
        super().changeEvent(e)

    def _scroll_to_bottom(self):
        QApplication.processEvents()
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

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

    def _add_emoji_bubble(self, emotion):
        w = EmojiResult(emotion)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(w)
        row.addStretch()
        self.chat_layout.addLayout(row)
        self._scroll_to_bottom()

    def _set_state(self, state):
        self.titlebar.set_state(state)

    def _finish_ui(self):
        self.button.setEnabled(True)
        self.button.setText("🎤  Record and Detect")
        self.is_processing = False
        self.sig_state.emit("idle")

    def _clean_transcript(self, s):
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
            # logger.info(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")

            calib_path = os.path.join(track["dir"], "calibration.json")
            calib = _read_json(calib_path, {})
            self.classes = calib.get("classes", ["happy", "sad"])

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

            if self.model_type == "ssl":
                global SSLFrontend
                if SSLFrontend is None:
                    from ssl_frontend import SSLFrontend as _SSL
                    SSLFrontend = _SSL
                self.ssl = SSLFrontend()

        except Exception:
            traceback.print_exc()
            self.sig_add_msg.emit("Setup problem: model could not be loaded.", False)

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

    def _say_blocking(self, text):
        """
        Blocking speak (called inside worker thread).
        Pepper if available, else pyttsx3.
        """
        try:
            if self.pepper:
                logger.info("[TTS] Pepper speaking…")
                self.pepper.tts(text)
            else:
                logger.info("[TTS] Laptop speaking…")
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            logger.info(f"[TTS] failed: {e}")

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
        self.sig_state.emit("listening")

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """
        One click = one run.
        Always ends with sig_finish so button resets.
        """
        try:
            target_sr = int(FEATURE_SETTINGS.get("sample_rate", 16000))
            dur = float(self.calib.get("record_seconds", 5.0))
            use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))

            logger.info(f"[REC] use_pepper={use_pepper} target_sr={target_sr} dur={dur:.1f}s")

            # -------- Record --------
            if use_pepper:
                try:
                    raw = self.pepper.record(seconds=int(max(1, round(dur))), mode=PEPPER.get("record_mode", "seconds"))
                    pepper_sr_hint = int(PEPPER.get("sample_rate", target_sr))
                    y, sr_file = _bytes_to_audio(raw, sr_hint=pepper_sr_hint)
                    if sr_file != target_sr:
                        y = librosa.resample(y, orig_sr=sr_file, target_sr=target_sr)
                    sr = target_sr
                except Exception as e:
                    logger.info(f"[REC] Pepper record failed -> laptop fallback ({e})")
                    y, sr = record_laptop(dur, target_sr)
            else:
                y, sr = record_laptop(dur, target_sr)

            # Clean once
            y = remove_fan_noise(y, sr)

            # Debug save
            if DEBUG_AUDIO_SAVE:
                ts = int(time.time())
                fname = f"debug_audio_{ts}.wav"
                sf.write(f"debug_audio_{ts}.wav", y, sr)
                logger.info(f"[DEBUG] Saved debug_audio_{ts}.wav")
                print(f"[DEBUG] Saved processed audio to: {fname}")

            peak = float(np.max(np.abs(y))) if y.size else 0.0
            rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
            logger.info(f"[AUDIO] peak={peak:.4f} rms={rms:.4f} samples={y.size}")

            if peak < 0.01:
                # ONE message only
                self.sig_add_msg.emit("🎤 (too quiet / no mic input)", True)
                self.sig_state.emit("idle")
                self._say_blocking("I couldn't hear you. Please try again.")
                return

            # -------- ASR --------
            self.sig_state.emit("thinking")
            raw_txt = self.asr.transcribe(y, sr)
            transcript = self._clean_transcript(raw_txt)
            logger.info(f"[ASR raw] {raw_txt!r}")
            logger.info(f"[ASR clean] {transcript!r}")

            if not transcript:
                # ONE message only
                self.sig_add_msg.emit("I heard sound, but no words.", False)
                self.sig_state.emit("idle")
                self._say_blocking("I heard sound, but no words.")
                return

            # show + log transcript
            self.sig_add_msg.emit(f"You: {transcript}", True)
            logger.info(f"[USER] {transcript}")

            # -------- Predict --------
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")
            logits = self.session.run(None, {self.input_name: x})[0][0]
            probs = softmax(logits / max(1e-6, float(self.temperature)))

            try:
                idx_h = self.classes.index("happy")
                idx_s = self.classes.index("sad")
            except ValueError:
                idx_h, idx_s = 0, 1

            prob_h = float(probs[idx_h])
            prob_s = float(probs[idx_s])
            logger.info(f"[MODEL] p(happy)={prob_h:.2f} p(sad)={prob_s:.2f}")

            audio_label = "happy" if prob_h > prob_s else "sad"

            # -------- Sentiment fusion (optional) --------
            final_label = audio_label
            if USE_SENTIMENT_FUSION:
                try:
                    polarity = TextBlob(transcript).sentiment.polarity
                    if polarity > SENTIMENT_POS_THRESHOLD:
                        final_label = "happy"
                        logger.info(f"[SENT] override HAPPY (polarity={polarity:.2f})")
                    elif polarity < SENTIMENT_NEG_THRESHOLD:
                        final_label = "sad"
                        logger.info(f"[SENT] override SAD (polarity={polarity:.2f})")
                except Exception:
                    pass

            self.sig_add_emoji.emit(final_label)

            # -------- Reply --------
            if self.dialog_phase == "opener":
                reply = RESPONSES.get(final_label, "I am listening.")
                self.dialog_phase = "chat"
            else:
                reply = self.chat.reply(final_label, transcript)

            self.sig_add_msg.emit(reply, False)
            logger.info(f"[BOT ] {reply}")

            # Speak reply (blocking in worker)
            self._say_blocking(reply)

        # except Exception as e:
        #     logger.info(f"[ERR] {e}")
        #     traceback.print_exc()
        #     self.sig_add_msg.emit("Error processing audio.", False)
        except Exception as e:
            # log("REC", f"recording failed -> {e}")
            self.sig_add_msg.emit("🎤 (recording failed)", True)
            self.sig_set_state.emit("idle")
            self._finish()     # reset button NOW
            return

        finally:
            self.sig_finish.emit()

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_icon_ico = os.path.join(base_dir, "SER_Mental_Health_Robot.ico")
    if os.path.exists(app_icon_ico):
        app.setWindowIcon(QIcon(app_icon_ico))

    w = EmotionAppFrameless()
    w.show()
    sys.exit(app.exec_())
