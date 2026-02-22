# gui_live_predict.py
import sys, os, json, threading, traceback, tempfile, time
import numpy as np
import librosa, sounddevice as sd, pyttsx3
import soundfile as sf
import requests
import onnxruntime as ort
from io import BytesIO
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
)
from extract_features import extract_mfcc

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES, PEPPER
try:
    from config import ASSISTANT_STYLE
except Exception:
    ASSISTANT_STYLE = (
        "You are a friendly, everyday wellbeing companion for mild support. "
        "Keep replies warm, natural, and brief (1–2 short sentences)(max 30 words). No diagnosis or clinical terms."
    )

try:
    from pepper_client import PepperClient
except Exception:
    PepperClient = None

# ==== Runtime tweaks (Windows OpenMP noise) ====
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ---------------- Files & loading ----------------
MODELS_DIR = "models"
TRACKS = [
    {"name": "SSL v1",  "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"],                           "type": "ssl"},
    {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], "type": "mfcc"},
]

# ---------------- Utils ----------------
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _speak_async(text, on_done):
    def run():
        try:
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
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
    """Return (y, sr) where y is float32 mono in [-1, 1]. Handles WAV or raw PCM16."""
    # WAV/AIFF?
    if len(raw) >= 12 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
        y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        return y, sr

    # Otherwise assume little-endian PCM16 mono
    a = np.frombuffer(raw, dtype=np.int16)
    if a.size == 0:
        raise ValueError("Pepper returned empty audio payload")
    y = (a.astype(np.float32)) / 32768.0
    return y, sr_hint


def _measure_level(y: np.ndarray):
    """Return (peak, rms) for the audio array."""
    if y is None or len(y) == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y))))
    return peak, rms


class LocalASR:
    def __init__(self, model_size="small"):  # try "small" instead of "base"
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",  # or "float32" if your CPU can handle it
        )

    def transcribe(self, y, sr):
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, y, sr, subtype="PCM_16")
            segments, _ = self.model.transcribe(
                path,
                language="en",
                task="transcribe",
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
            )
            text = " ".join(s.text for s in segments).strip()
            return text or None
        finally:
            try:
                os.remove(path)
            except Exception:
                pass




# ---------------- LLM via Ollama (chat -> generate fallback) ----------------
class ChatEngine:
    """
    Works with both newer (/api/chat) and older (/api/generate) Ollama servers.
    Pull a model:  ollama pull llama3.1:8b-instruct
    """
    def __init__(self, model=None, host=None, debug=True):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.url_chat = f"{self.host}/api/chat"
        self.url_gen = f"{self.host}/api/generate"
        self.system = ASSISTANT_STYLE
        self.history = []
        self.debug = debug
        try:
            ok = requests.get(f"{self.host}/api/tags", timeout=5).ok
            print(f"[Ollama] host={self.host} online={ok}")
        except Exception as e:
            print(f"[Ollama] ping failed: {e}")

    def _make_prompt(self, emotion: str, transcript: str | None) -> str:
        return (
            f"Emotion={emotion or 'unknown'}\n"
            f"Transcript={transcript or '(empty)'}\n"
            "Respond in 1–2 short(max 30 words), natural sentences. Avoid clinical terms."
        )

    def _messages(self, prompt: str):
        return [
            {"role": "system", "content": self.system},
            *self.history[-8:],
            {"role": "user", "content": prompt},
        ]

    def _compose_text_prompt(self, prompt: str) -> str:
        # Convert system+history into a single prompt for /api/generate
        parts = [f"System: {self.system}"]
        for m in self.history[-8:]:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def reply(self, emotion: str, transcript: str | None) -> str:
        prompt = self._make_prompt(emotion, transcript)

        # Try /api/chat
        try:
            print(f"[Ollama] POST {self.url_chat} model={self.model}")
            r = requests.post(
                self.url_chat,
                json={
                    "model": self.model,
                    "messages": self._messages(prompt),
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "repeat_penalty": 1.15,
                        "num_predict": 160,
                    },
                },
                timeout=20,
            )
            if r.status_code == 404:
                raise requests.HTTPError("404 chat endpoint", response=r)
            r.raise_for_status()
            text = (r.json().get("message", {}) or {}).get("content", "")
            if text.strip():
                self.history += [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ]
                self.history = self.history[-12:]
                return text.strip()
            raise RuntimeError("empty chat response")
        except Exception as e:
            if self.debug:
                print("[Ollama chat error]", repr(e))

        # Fallback to /api/generate
        try:
            text_prompt = self._compose_text_prompt(prompt)
            print(f"[Ollama] POST {self.url_gen} model={self.model}")
            r = requests.post(
                self.url_gen,
                json={
                    "model": self.model,
                    "prompt": text_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "repeat_penalty": 1.15,
                        "num_predict": 160,
                    },
                },
                timeout=20,
            )
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            if not text:
                raise RuntimeError("empty generate response")
            self.history += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ]
            self.history = self.history[-12:]
            return text
        except Exception as e:
            if self.debug:
                print("[Ollama generate error]", repr(e))
            return (
                "I’m having a hiccup reaching my language model. "
                "We can keep talking, or try again shortly."
            )


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
    sig_add_msg = pyqtSignal(str, bool)   # text, is_user
    sig_finish = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Speech Emotion Detection Application")
        self.setGeometry(100, 100, 400, 500)
        self.emotion_locked = None
        self.emotion_locked_at = None
        self.turns_since_lock = 0

        # top bar
        self.mic_icon = QLabel()
        self.mic_off = QPixmap("mic-off.png").scaled(
            24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.mic_on = QPixmap("mic-on.png").scaled(
            24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.mic_icon.setPixmap(self.mic_off)

        # Chat area
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        container = QWidget()
        container.setLayout(self.chat_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.scroll = scroll

        # Record button (single control)
        self.button = QPushButton("Record & Detect")
        self.button.setIcon(QIcon("mic-off.png"))
        self.button.setIconSize(QSize(20, 20))
        self.button.clicked.connect(self.record_and_predict)

        # Layout
        main = QVBoxLayout(self)
        main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
        main.addWidget(scroll)
        main.addWidget(self.button)

        # Runtime state
        self.is_processing = False
        self.session = None
        self.input_name = None
        self.model_type = None       # "ssl" or "mfcc"
        self.classes = ["happy", "sad"]
        self.calib = {
            "mode": "threshold",
            "sad_threshold": 0.57,
            "min_confidence": 0.50,
            "min_amp": 0.10,
            "record_seconds": 3.0,
        }
        self.temperature = 1.0
        self.ssl = None              # SSLFrontend (lazy)

        # Conversation engines/state
        self.asr = LocalASR()
        self.chat = ChatEngine(debug=True)
        self.dialog_phase = "opener"   # first turn uses RESPONSES[label], then LLM
        self.sig_add_msg.connect(self._add_msg)     # ensure _add_msg runs on GUI thread
        self.sig_finish.connect(self._finish_ui)    # ensure UI reset runs on GUI thread

        self._auto_load_model()
        self.pepper = None
        if PEPPER.get("enabled") and PepperClient:
            try:
                self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
                self.pepper.connect()
                print("[Pepper] connected")
            except Exception as e:
                print("[Pepper] connect failed:", e)
                self.pepper = None

    # --- Emotion lock helpers ---
    def _should_decay_lock(self, minutes=7, max_turns=4):
        if self.emotion_locked is None:
            return True
        if self.emotion_locked_at and (time.time() - self.emotion_locked_at) > minutes * 60:
            return True
        if self.turns_since_lock >= max_turns:
            return True
        return False

    def _lock_emotion(self, label, probs=None):
        """Lock only when reasonably sure, or always if probs is None (self-report override)."""
        if probs is not None:
            try:
                idx = self.classes.index(label)
                if float(probs[idx]) < 0.75:
                    # Don't lock on low-confidence prediction
                    return
            except ValueError:
                pass
        self.emotion_locked = label
        self.emotion_locked_at = time.time()
        self.turns_since_lock = 0

    def _say(self, text):
        """
        Speak without changing UI state.
        The worker calls self._finish() AFTER this returns.
        """
        if self.pepper:
            # 👉 BLOCKING: wait until Pepper finishes speaking
            try:
                self.pepper.tts(text)  # NAOqi call, runs in this worker thread
            except Exception:
                traceback.print_exc()
        else:
            # Local PC TTS can stay async
            _speak_async(text, lambda: None)




    def _maybe_override_from_text(self, transcript: str | None):
        if not transcript:
            return
        t = transcript.lower()
        if any(p in t for p in ["i'm happy", "i am happy", "feeling happy"]):
            self._lock_emotion("happy")
        elif any(p in t for p in ["i'm sad", "i am sad", "feeling sad", "feeling down", "upset"]):
            self._lock_emotion("sad")

    # ---- Robot integration hook (use later) ----
    def on_prediction(self, label: str):
        pass

    # ---- Minimal chat helpers ----
    def _add_msg(self, text, is_user=False):
        bubble = ChatBubble(text, is_user)
        row = QHBoxLayout()
        if is_user:
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()
        f = QFrame()
        f.setLayout(row)
        self.chat_layout.addWidget(f)
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _add_msg_safe(self, text, is_user=False):
        self.sig_add_msg.emit(text, is_user)

    # ---- Model load (auto, silent) ----
    def _auto_load_model(self):
        try:
            chosen = None
            for t in TRACKS:
                tdir = t["dir"]
                onnx_path = _find_existing([os.path.join(tdir, fn) for fn in t["onnx"]])
                if onnx_path:
                    chosen = (t, onnx_path)
                    break
            if not chosen:
                raise FileNotFoundError(
                    "No ONNX model found in models/ssl_v1 or models/mfcc_v1."
                )

            track, onnx_path = chosen
            self.model_type = track["type"]
            print(
                f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}"
            )

            # load per-track calibration + optional temperature
            calib_path = os.path.join(track["dir"], "calibration.json")
            self.calib.update(_read_json(calib_path, {}))
            temp = _read_json(
                os.path.join(track["dir"], "temperature.json"), {"temperature": 1.0}
            )
            self.temperature = float(temp.get("temperature", 1.0))

            # classes order (optional in calibration)
            self.classes = _read_json(calib_path, {}).get("classes", self.classes)

            # ORT session
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            self.session = ort.InferenceSession(
                onnx_path, so, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name

            # lazy SSL frontend if needed
            if self.model_type == "ssl" and self.ssl is None:
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

        except Exception:
            traceback.print_exc()
            self._add_msg_safe("Setup problem. Please check model files.", is_user=False)

    # ---- Features (private) ----
    def _feat_mfcc(self, y, sr):
        target = FEATURE_SETTINGS.get("sample_rate", 16000)
        if sr != target:
            y = librosa.resample(y, orig_sr=sr, target_sr=target)
            sr = target
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
        try:
            yt, _ = librosa.effects.trim(y, top_db=30)
            if len(yt) > int(0.25 * sr):
                y = yt
        except Exception:
            pass
        return self.ssl(y)

    # ---- Decoding (silent; no probs shown) ----
    def _decode(self, probs):
        probs = np.asarray(probs, dtype=float)
        try:
            idx_h = self.classes.index("happy")
            idx_s = self.classes.index("sad")
        except ValueError:
            idx_h, idx_s = 0, 1

        p_h = float(probs[idx_h])
        p_s = float(probs[idx_s])
        p_max = max(p_h, p_s)
        margin = abs(p_h - p_s)

        # Strengthen confidence requirement a bit
        min_conf = float(self.calib.get("min_confidence", 0.50))
        # min_conf = max(min_conf, 0.60)  # enforce at least 0.6

        if p_max < min_conf or margin < 0.05:
            return "Uncertain"

        mode = self.calib.get("mode", "threshold")
        sad_thr = float(self.calib.get("sad_threshold", 0.57))
        # clamp weird thresholds (e.g., old 1.4 bug) into [0, 1]
        sad_thr = min(max(sad_thr, 0.0), 1.0)

        if mode == "threshold":
            return "sad" if p_s >= sad_thr else "happy"
        return "happy" if p_h >= p_s else "sad"

    def _finish_ui(self):
        self.mic_icon.setPixmap(self.mic_off)
        self.button.setText("🎤  Record & Detect")
        self.button.setIcon(QIcon("mic-off.png"))
        self.button.setEnabled(True)
        self.is_processing = False

    def _finish(self):
        # safe to call from any thread (worker / TTS thread)
        self.sig_finish.emit()

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

        # run the heavy pipeline off the UI thread
        threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

    def _record_and_predict_worker(self):
        try:
            # --- record (short and bounded) ---
            sr = FEATURE_SETTINGS.get("sample_rate", 16000)
            dur = float(self.calib.get("record_seconds", 3.0))
            use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
            print(f"[Audio] use_pepper_mic={use_pepper}")

            if use_pepper:
                try:
                    raw = self.pepper.record(
                        seconds=int(max(1, round(dur))),
                        mode=PEPPER.get("record_mode", "auto"),
                    )
                    y, sr_file = _bytes_to_audio(
                        raw, sr_hint=int(PEPPER.get("sample_rate", 48000))
                    )
                    sf.write("debug_pepper.wav", y, sr_file)
                    print("[DEBUG] saved debug_pepper.wav with sr_file =", sr_file)
                    if sr_file != sr:
                        y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
                    print("[Audio] source: Pepper mic")
                except Exception as e:
                    traceback.print_exc()
                    print("[Audio] Pepper record failed, falling back to PC mic:", e)
                    y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
                    sd.wait()
                    y = y.flatten()
                    print("[Audio] source: PC mic")
            else:
                y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                y = y.flatten()
                print("[Audio] source: PC mic")

            # --------------------------------------------------
            # 1) TWO-STAGE GATE: "any audio?" and "speech strong enough?"
            # --------------------------------------------------
            peak, rms = _measure_level(y)

            if use_pepper:
                 # Stage-1: very quiet background → treat as no speech at all
                basic_min_peak = float(self.calib.get("basic_min_peak", 0.03))
                basic_min_rms  = float(self.calib.get("basic_min_rms", 0.004))

                # Stage-2: "speech strong enough to trust emotion"
                # Your normal speech in logs is around peak 0.17–0.35, rms 0.015–0.03,
                # so we set these a bit lower than that.
                speech_min_peak = float(self.calib.get("speech_min_peak", 0.12))
                speech_min_rms  = float(self.calib.get("speech_min_rms", 0.008))
            else:
                # PC mic can be more sensitive
                basic_min_peak = float(self.calib.get("basic_min_peak", 0.02))
                basic_min_rms  = float(self.calib.get("basic_min_rms", 0.003))
                speech_min_peak = float(self.calib.get("speech_min_peak", 0.10))
                speech_min_rms  = float(self.calib.get("speech_min_rms", 0.010))

            print(
                f"[Gate] peak={peak:.3f} rms={rms:.4f} "
                f"(basic_min_peak={basic_min_peak:.3f}, basic_min_rms={basic_min_rms:.4f}, "
                f"speech_min_peak={speech_min_peak:.3f}, speech_min_rms={speech_min_rms:.4f})"
            )

            # Stage 1: basically silence -> don't do SER at all
            if peak < basic_min_peak or rms < basic_min_rms:
                msg = "I couldn’t hear you clearly just now. Could you try a bit closer to the mic?"
                self._add_msg_safe("🎤 (no clear speech)", is_user=False)
                self._add_msg_safe(msg, is_user=False)
                self._say(msg)
                self._finish() 
                return

            # We heard something: mark that on the right side
            self._add_msg_safe("🎤 (audio captured)", is_user=True)

            # --- features + SER inference ---
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")
            logits = self.session.run(None, {self.input_name: x})[0][0]
            T = max(1e-6, float(self.temperature))
            probs = softmax(logits / T)
            label = self._decode(probs)
            print(f"[SER] probs={probs} -> label={label}")

            # Stage 2: audio was there but not strong/clear enough to trust emotion
            # if peak < speech_min_peak and rms < speech_min_rms:
            #     msg = (
            #         "I heard a little sound, but not clearly enough to notice how you might be "
            #         "feeling. Could you try speaking a bit closer or louder?"
            #     )
            #     self._add_msg_safe(msg, is_user=False)
            #     self._say(msg)
            #     self._finish()
            #     return

            # SER ran but is still uncertain about emotion
            if label == "Uncertain":
                msg = RESPONSES.get(
                    "Uncertain",
                    "I’m not quite sure how you’re feeling right now. Would you like to try again?",
                )
                self._add_msg_safe(msg, is_user=False)
                self._say(msg)
                self._finish()
                return

            # --- ASR (Whisper) ---
            transcript = None
            try:
                transcript = self.asr.transcribe(y, sr)
            except Exception:
                traceback.print_exc()
                transcript = None

            print(f"[ASR] raw transcript={transcript!r}")

            if transcript:
                clean = transcript.strip()
                if len(clean) < 2:
                    print("[ASR] transcript discarded as too short")
                    transcript = None

            print(f"[ASR] final transcript={transcript!r}")

            # Show transcript / note in the chat
            if not transcript:
                # self._add_msg_safe("🎤 (no transcript)", is_user=False)
                msg = (
                    "I heard your voice, but I couldn’t quite catch the words. "
                    "Could you please repeat that a bit closer to the mic?"
                )
                self._add_msg_safe(msg, is_user=False)
                self._say(msg)
                self._finish()
                return

         
            self._add_msg_safe(transcript, is_user=True)
            
            
            # --- Emotion lock + text override ---
            if self._should_decay_lock() and label in ("happy", "sad"):
                self._lock_emotion(label, probs=probs)

            self._maybe_override_from_text(transcript)

            # Emotion we send to LLM
            emotion_for_llm = self.emotion_locked or (
                label if label in ("happy", "sad") else "unknown"
            )


            # --- Generate reply ---
            if self.dialog_phase == "opener":
                reply = RESPONSES.get(
                    emotion_for_llm,
                    RESPONSES.get(
                        "Uncertain",
                        "I am not sure how you are feeling. Would you like to try again.",
                    ),
                )
                self.dialog_phase = "chat"
            else:
                reply = self.chat.reply(emotion_for_llm, transcript)

            self.turns_since_lock += 1
            self._add_msg_safe(reply, is_user=False)
            self._say(reply)
            self.on_prediction(label)
            debug_path = "debug_pepper.wav"
            try:
                sf.write(debug_path, y, sr, subtype="PCM_16")
                print(f"[DEBUG] wrote {debug_path}, shape={y.shape}, sr={sr}")
            except Exception as e:
                print("[DEBUG] failed to write debug_pepper.wav:", e)
            # self.on_prediction(label)
            # self._say(reply)
            self._finish()

        except KeyboardInterrupt:
            self._add_msg_safe("⏹️ cancelled.", is_user=False)
            self._finish()
        except Exception:
            traceback.print_exc()
            self._add_msg_safe("Something went wrong. Let's try again.", is_user=False)
            self._finish()


    def reset_session(self):
        self.emotion_locked = None
        self.emotion_locked_at = None
        self.turns_since_lock = 0
        self.chat.history = []
        self.dialog_phase = "opener"
        self._add_msg_safe("🆕 New session started.", is_user=False)


# ---- Run ----
if __name__ == "__main__":
    print("[BOOT] OLLAMA_HOST =", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    print("[BOOT] OLLAMA_MODEL =", os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
    app = QApplication(sys.argv)
    w = EmotionApp()
    w.show()
    sys.exit(app.exec_())
