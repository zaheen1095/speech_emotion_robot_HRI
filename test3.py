# # gui_live_predict.py
# import sys, os, json, threading, traceback, tempfile, time
# import numpy as np
# import librosa, sounddevice as sd, pyttsx3
# import soundfile as sf
# import requests
# import onnxruntime as ort
# from io import BytesIO
# from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
# from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QPushButton, QLabel,
#     QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
# )
# from extract_features import extract_mfcc

# # --- Project modules ---
# from config import FEATURE_SETTINGS, RESPONSES, PEPPER
# try:
#     from config import ASSISTANT_STYLE
# except Exception:
#     ASSISTANT_STYLE = (
#         "You are a friendly, everyday wellbeing companion for mild support. "
#         "Keep replies warm, natural, and brief (1–2 short sentences). No diagnosis or clinical terms."
#     )
#     # ASSISTANT_STYLE = (
#     #     "You are a gentle, everyday wellbeing companion for mild emotional support only. "
#     #     "Sometimes the user feels happy or excited, sometimes they feel a bit sad, low, worried, or stressed. "
#     #     "You are NOT a doctor, therapist, or crisis service.\n"
#     #     "General guidelines:\n"
#     #     "- Keep replies warm, simple, and human (1–2 short sentences).\n"
#     #     "- Never use clinical or diagnostic language (no 'disorder', 'diagnosis', 'treatment plan').\n"
#     #     "- Never mention medication or give medical advice.\n"
#     #     "- If the user sounds very distressed or talks about self-harm, gently encourage them to reach out "
#     #     "to a trusted person or professional, and keep your tone calm and supportive.\n"
#     #     "- Ask simple, open questions sometimes (like 'Want to tell me a bit more?' or "
#     #     "'What usually helps you even a tiny bit?'), but never pressure them.\n"
#     #     "\n"
#     #     "When the user seems HAPPY:\n"
#     #     "- Reflect their good feeling and celebrate it a little.\n"
#     #     "- Invite them to notice what is going well or what helped them feel this way.\n"
#     #     "- You can suggest small ideas to keep the good feeling going (sharing with a friend, doing more "
#     #     "of what helped, saving the moment as a memory).\n"
#     #     "\n"
#     #     "When the user seems SAD or LOW:\n"
#     #     "- Validate their feeling and normalise that ups and downs are part of life.\n"
#     #     "- Offer 1–2 small, doable coping ideas (breathing, short walk, grounding, talking to someone they trust, "
#     #     "doing one tiny kind action for themselves).\n"
#     # )


# try:
#     from pepper_client import PepperClient
# except Exception:
#     PepperClient = None

# # ==== Runtime tweaks (Windows OpenMP noise) ====
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# # ---------------- Files & loading ----------------
# MODELS_DIR = "models"
# TRACKS = [
#     {"name": "SSL v1",  "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"],                           "type": "ssl"},
#     {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], "type": "mfcc"},
# ]

# # ---------------- Utils ----------------
# def softmax(x):
#     x = x - np.max(x)
#     e = np.exp(x)
#     return e / e.sum()


# def _speak_async(text, on_done):
#     def run():
#         try:
#             eng = pyttsx3.init()
#             eng.say(text)
#             eng.runAndWait()
#         except Exception:
#             traceback.print_exc()
#         finally:
#             QTimer.singleShot(0, on_done)
#     threading.Thread(target=run, daemon=True).start()


# def _read_json(path, default):
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return default


# def _find_existing(paths):
#     for p in paths:
#         if os.path.exists(p):
#             return p
#     return None


# def _bytes_to_audio(raw: bytes, sr_hint: int = 16000):
#     """Return (y, sr) where y is float32 mono in [-1, 1]. Handles WAV or raw PCM16."""
#     # WAV/AIFF?
#     if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
#         y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
#         if y.ndim > 1:
#             y = y[:, 0]
#         return y, sr

#     # Otherwise assume little-endian PCM16 mono
#     a = np.frombuffer(raw, dtype=np.int16)
#     if a.size == 0:
#         raise ValueError("Pepper returned empty audio payload")
#     y = (a.astype(np.float32)) / 32768.0
#     return y, sr_hint


# class LocalASR:
#     def __init__(self, model_size="small"):
#         from faster_whisper import WhisperModel
#         # CPU + int8 is fine, just slower
#         self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

#     def transcribe(self, y, sr):
#         """
#         Run faster-whisper on a temp WAV file.
#         Returns a raw string or None.
#         """
#         y = np.asarray(y, dtype=np.float32).reshape(-1)
#         if y.size == 0:
#             print("[ASR] empty audio")
#             return None

#         # --- make sure audio is 16 kHz for Whisper ---
#         if sr != 16000:
#             print(f"[ASR] resampling {sr} -> 16000 for Whisper")
#             y = librosa.resample(y, orig_sr=sr, target_sr=16000)
#             sr = 16000

#         peak = float(np.max(np.abs(y)))
#         duration = len(y) / sr
#         print(f"[ASR] dur={duration:.2f}s, sr={sr}, peak={peak:.4f}")

#         # light normalize
#         if peak > 0:
#             y = (y / peak) * 0.9

#         # small padding to avoid edge-cut
#         pad = np.zeros(int(0.10 * sr), dtype=np.float32)
#         y = np.concatenate([pad, y, pad], axis=0)

#         # --- write to temp wav & call Whisper ---
#         fd, path = tempfile.mkstemp(suffix=".wav")
#         os.close(fd)
#         try:
#             sf.write(path, y, sr, subtype="PCM_16")

#             segments, info = self.model.transcribe(
#                 path,
#                 language="en",
#                 vad_filter=True,   # let our own gate handle silence
#                 vad_parameters=dict(min_silence_duration_ms=500),
#                 beam_size=1,        # can increase to 3–5 if CPU is okay
#             )

#             texts = []
#             for s in segments:
#                 print(f"[ASR seg] {s.start:.2f}-{s.end:.2f}s: {s.text!r}")
#                 texts.append(s.text)

#             text = " ".join(texts).strip()
#             if not text:
#                 print("[ASR] empty transcript from Whisper")
#                 return None

#             print(f"[ASR raw text] {text!r}")
#             return text

#         except Exception as e:
#             print("[ASR ERROR]", repr(e))
#             traceback.print_exc()
#             return None

#         finally:
#             try:
#                 os.remove(path)
#             except Exception:
#                 pass



# class ChatEngine:
#     """
#     Works with both newer (/api/chat) and older (/api/generate) Ollama servers.
#     Pull a model:  ollama pull llama3.1:8b-instruct
#     """
#     def __init__(self, model=None, host=None, debug=True):
#         self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#         self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
#         self.url_chat = f"{self.host}/api/chat"
#         self.url_gen = f"{self.host}/api/generate"
#         self.system = ASSISTANT_STYLE
#         self.history = []
#         self.debug = debug
#         try:
#             ok = requests.get(f"{self.host}/api/tags", timeout=5).ok
#             print(f"[Ollama] host={self.host} online={ok}")
#         except Exception as e:
#             print(f"[Ollama] ping failed: {e}")

#     def _make_prompt(self, emotion: str, transcript: str | None) -> str:
#         t = (transcript or "").strip()
#         return (
#             f"{self.system}\n"
#             f"UserEmotion: {emotion or 'unknown'}\n"
#             f"UserSaid: {t if t else '(empty)'}\n"
#             "If UserSaid looks like a question (contains '?' or starts with: how, what, why, can, should, could), "
#             "answer it directly with 1–2 concrete suggestions. "
#             "Otherwise reply in 1–2 short, natural sentences.\n"
#             "Be specific and actionable."
#         )

#     def _messages(self, prompt: str):
#         return [
#             {"role": "system", "content": self.system},
#             *self.history[-8:],
#             {"role": "user", "content": prompt},
#         ]

#     def _compose_text_prompt(self, prompt: str) -> str:
#         # Convert system+history into a single prompt for /api/generate
#         parts = [f"System: {self.system}"]
#         for m in self.history[-8:]:
#             parts.append(f"{m['role'].capitalize()}: {m['content']}")
#         parts.append(f"User: {prompt}")
#         parts.append("Assistant:")
#         return "\n".join(parts)

#     def reply(self, emotion: str, transcript: str | None) -> str:
#         prompt = self._make_prompt(emotion, transcript)

#         # --- single fast call to /api/generate ---
#         try:
#             text_prompt = self._compose_text_prompt(prompt)
#             print(f"[Ollama] POST {self.url_gen} model={self.model}")
#             r = requests.post(
#                 self.url_gen,
#                 json={
#                     "model": self.model,
#                     "prompt": text_prompt,
#                     "stream": False,
#                     "options": {
#                         "temperature": 0.6,
#                         "repeat_penalty": 1.15,
#                         "num_predict": 80,  # short reply
#                     },
#                 },
#                 timeout=15,
#             )
#             r.raise_for_status()
#             text = (r.json().get("response") or "").strip()
#             if not text:
#                 raise RuntimeError("empty generate response")

#             self.history += [
#                 {"role": "user", "content": prompt},
#                 {"role": "assistant", "content": text},
#             ]
#             self.history = self.history[-12:]
#             return text

#         except Exception as e:
#             if self.debug:
#                 print("[Ollama generate error]", repr(e))
#             return (
#                 "I’m having a hiccup reaching my language model. "
#                 "We can keep talking, or try again shortly."
#             )


# # ---------------- Chat UI ----------------
# class ChatBubble(QLabel):
#     def __init__(self, text, is_user=False):
#         super().__init__(text)
#         self.setWordWrap(True)
#         self.setMaximumWidth(360)
#         color = "#e0e0e0" if is_user else "#95abbe"
#         self.setStyleSheet(f"background:{color}; border-radius:10px; padding:8px;")


# # ---------------- Main App ----------------
# class EmotionApp(QWidget):
#     sig_add_msg = pyqtSignal(str, bool)  # text, is_user
#     sig_finish = pyqtSignal()

#     def __init__(self):
#         super().__init__()
#         self._last_click = 0.0
#         self.setWindowTitle("Speech Emotion Detection Application")
#         self.setGeometry(100, 100, 400, 500)
#         self.emotion_locked = None
#         self.emotion_locked_at = None
#         self.turns_since_lock = 0

#         # top bar
#         self.mic_icon = QLabel()
#         self.mic_off = QPixmap("mic-off.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_on = QPixmap("mic-on.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_icon.setPixmap(self.mic_off)

#         # Chat area
#         self.chat_layout = QVBoxLayout()
#         self.chat_layout.setAlignment(Qt.AlignTop)
#         container = QWidget()
#         container.setLayout(self.chat_layout)
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)
#         scroll.setWidget(container)
#         self.scroll = scroll

#         # Record button (single control)
#         self.button = QPushButton("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setIconSize(QSize(20, 20))
#         self.button.clicked.connect(self.record_and_predict)

#         # Layout
#         main = QVBoxLayout(self)
#         main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
#         main.addWidget(scroll)
#         main.addWidget(self.button)

#         # Runtime state
#         self.is_processing = False
#         self.session = None
#         self.input_name = None
#         self.model_type = None  # "ssl" or "mfcc"
#         self.classes = ["happy", "sad"]
#         self.calib = {
#             "mode": "threshold",
#             "sad_threshold": 0.57,
#             "min_confidence": 0.50,
#             "min_amp": 0.10,
#             "record_seconds": 3.0,
#         }
#         self.temperature = 1.0
#         self.ssl = None  # SSLFrontend (lazy)

#         # Conversation engines/state
#         self.asr = LocalASR()
#         self.chat = ChatEngine(debug=True)
#         self.dialog_phase = "opener"  # first turn uses RESPONSES[label], then LLM
#         self.sig_add_msg.connect(self._add_msg)      # GUI thread
#         self.sig_finish.connect(self._finish_ui)     # GUI thread

#         self._auto_load_model()
#         self.pepper = None
#         if PEPPER.get("enabled") and PepperClient:
#             try:
#                 self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
#                 self.pepper.connect()
#                 print("[Pepper] connected")
#             except Exception as e:
#                 print("[Pepper] connect failed:", e)
#                 self.pepper = None

#     # --- Speech gate ---
#     def _has_speech(self, y, sr):
#         if y is None:
#             return False
#         y = np.asarray(y, dtype=np.float32)
#         if y.size == 0:
#             return False

#         # Drop Pepper relay / click at start just for stats
#         drop = int(0.25 * sr)
#         if y.size > drop:
#             y = y[drop:]

#         # Basic stats
#         peak = float(np.max(np.abs(y)))
#         rms  = float(np.sqrt(np.mean(np.square(y))))
#         frame = max(1, int(0.030 * sr))
#         hop   = max(1, int(0.015 * sr))
#         zcr   = float(
#             librosa.feature.zero_crossing_rate(
#                 y, frame_length=frame, hop_length=hop
#             ).mean()
#         )
#         p95 = float(np.percentile(np.abs(y), 95))
#         loud_ratio = float(np.mean(np.abs(y) > 0.02))

#         # Much looser thresholds – rely on min_amp + ASR to filter junk
#         # peak_th = 0.03
#         # rms_th  = 0.006
#         # p95_th  = 0.015
#         # zcr_th  = 0.020
#         # loud_th = 0.01

#         # --- TUNABLE THRESHOLDS (looser) ---
#         peak_th = 0.05     # keep
#         rms_th  = 0.010    # a bit looser
#         p95_th  = 0.030    # was 0.040
#         loud_th = 0.070    # was 0.16
#         zcr_th  = 0.020    # a bit looser

#         # # --- TUNABLE THRESHOLDS (Low Sensitivity - Fixes BLOCK issues) ---
#         # peak_th = 0.04      # Very low
#         # rms_th  = 0.008     # <--- CRITICAL FIX (Was blocking at 0.016)
#         # p95_th  = 0.020     
#         # loud_th = 0.050     # <--- CRITICAL FIX (Was blocking at 0.113)
#         # zcr_th  = 0.015


#         passed = (
#             peak >= peak_th
#             and rms  >= rms_th
#             and p95  >= p95_th
#             and zcr  >= zcr_th
#             and loud_ratio >= loud_th
#         )

#         print(
#             f"[Gate] peak={peak:.4f} rms={rms:.4f} zcr={zcr:.4f} "
#             f"p95={p95:.4f} loud={loud_ratio:.3f} -> {'PASS' if passed else 'BLOCK'}"
#         )
#         return passed


#     def _should_decay_lock(self, minutes=7, max_turns=4):
#         if self.emotion_locked is None:
#             return True
#         if self.emotion_locked_at and (time.time() - self.emotion_locked_at) > minutes * 60:
#             return True
#         if self.turns_since_lock >= max_turns:
#             return True
#         return False

#     def _lock_emotion(self, label):
#         self.emotion_locked = label
#         self.emotion_locked_at = time.time()
#         self.turns_since_lock = 0

#     def _say(self, text):
#         def finish_on_ui():
#             QTimer.singleShot(0, self._finish)  # reset is_processing/button on GUI thread

#         if self.pepper:
#             def run():
#                 try:
#                     self.pepper.tts(text)  # blocking call in this worker thread
#                 except Exception:
#                     traceback.print_exc()
#                 finally:
#                     finish_on_ui()
#             threading.Thread(target=run, daemon=True).start()
#         else:
#             _speak_async(text, finish_on_ui)

#     def _maybe_override_from_text(self, transcript: str | None):
#         if not transcript:
#             return
#         t = transcript.lower()
#         if any(p in t for p in ["i'm happy", "i am happy", "feeling happy"]):
#             self._lock_emotion("happy")
#         elif any(
#             p in t
#             for p in ["i'm sad", "i am sad", "feeling sad", "feeling down", "upset"]
#         ):
#             self._lock_emotion("sad")

#     # ---- Robot integration hook (use later) ----
#     def on_prediction(self, label: str):
#         pass

#     # ---- Minimal chat helpers ----
#     def _add_msg(self, text, is_user=False):
#         bubble = ChatBubble(text, is_user)
#         row = QHBoxLayout()
#         if is_user:
#             row.addStretch()
#             row.addWidget(bubble)
#         else:
#             row.addWidget(bubble)
#             row.addStretch()
#         f = QFrame()
#         f.setLayout(row)
#         self.chat_layout.addWidget(f)
#         sb = self.scroll.verticalScrollBar()
#         sb.setValue(sb.maximum())

#     def _add_msg_safe(self, text, is_user=False):
#         self.sig_add_msg.emit(text, is_user)

#     # ---- Model load (auto, silent) ----
#     def _auto_load_model(self):
#         try:
#             chosen = None
#             for t in TRACKS:
#                 tdir = t["dir"]
#                 onnx_path = _find_existing([os.path.join(tdir, fn) for fn in t["onnx"]])
#                 if onnx_path:
#                     chosen = (t, onnx_path)
#                     break
#             if not chosen:
#                 raise FileNotFoundError(
#                     "No ONNX model found in models/ssl_v1 or models/mfcc_v1."
#                 )

#             track, onnx_path = chosen
#             self.model_type = track["type"]
#             print(
#                 f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}"
#             )

#             # load per-track calibration + optional temperature
#             calib_path = os.path.join(track["dir"], "calibration.json")
#             self.calib.update(_read_json(calib_path, {}))
#             temp = _read_json(
#                 os.path.join(track["dir"], "temperature.json"), {"temperature": 1.0}
#             )
#             self.temperature = float(temp.get("temperature", 1.0))

#             # classes order (optional in calibration)
#             self.classes = _read_json(calib_path, {}).get("classes", self.classes)

#             # ORT session
#             so = ort.SessionOptions()
#             so.intra_op_num_threads = 1
#             so.inter_op_num_threads = 1
#             so.log_severity_level = 3
#             self.session = ort.InferenceSession(
#                 onnx_path, so, providers=["CPUExecutionProvider"]
#             )
#             self.input_name = self.session.get_inputs()[0].name

#             # lazy SSL frontend if needed
#             if self.model_type == "ssl" and self.ssl is None:
#                 import warnings

#                 try:
#                     from transformers.utils import logging as hf_logging

#                     hf_logging.set_verbosity_error()
#                 except Exception:
#                     pass
#                 warnings.filterwarnings(
#                     "ignore", message="Passing `gradient_checkpointing`.*"
#                 )
#                 warnings.filterwarnings(
#                     "ignore", message="`clean_up_tokenization_spaces`.*"
#                 )
#                 from ssl_frontend import SSLFrontend

#                 self.ssl = SSLFrontend()  # let your class pick model & device

#         except Exception:
#             traceback.print_exc()
#             self._add_msg_safe(
#                 "Setup problem. Please check model files.", is_user=False
#             )

#     # ---- Features (private) ----
#     def _feat_mfcc(self, y, sr):
#         target = FEATURE_SETTINGS.get("sample_rate", 16000)
#         if sr != target:
#             y = librosa.resample(y, orig_sr=sr, target_sr=target)
#             sr = target
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr):
#                 y = yt
#         except Exception:
#             pass
#         return extract_mfcc(array=y, sr=sr)

#     def _feat_ssl(self, y, sr):
#         if sr != 16000:
#             y = librosa.resample(y, orig_sr=sr, target_sr=16000)
#             sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr):
#                 y = yt
#         except Exception:
#             pass
#         return self.ssl(y)

#     # ---- Decoding (silent; no probs shown) ----
#     def _decode(self, probs):
#         try:
#             idx_h = self.classes.index("happy")
#             idx_s = self.classes.index("sad")
#         except ValueError:
#             idx_h, idx_s = 0, 1
#         p_h, p_s = float(probs[idx_h]), float(probs[idx_s])
#         p_max = max(p_h, p_s)
#         if p_max < float(self.calib.get("min_confidence", 0.50)):
#             return "Uncertain"
#         if self.calib.get("mode", "threshold") == "threshold":
#             return "sad" if p_s >= float(self.calib.get("sad_threshold", 0.57)) else "happy"
#         return "happy" if p_h >= p_s else "sad"

#     def _finish_ui(self):
#         self.mic_icon.setPixmap(self.mic_off)
#         self.button.setText("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setEnabled(True)
#         self.is_processing = False

#     def _finish(self):
#         # safe to call from any thread (worker / TTS thread)
#         self.sig_finish.emit()

#     # ---- Main action ----
#     def record_and_predict(self):
#         now = time.monotonic()
#         if now - self._last_click < 0.8:
#             return
#         self._last_click = now
#         if self.is_processing or self.session is None:
#             return
#         self.is_processing = True
#         self.button.setEnabled(False)
#         self.button.setText("🔴  Listening…")
#         self.button.setIcon(QIcon("mic-on.png"))
#         self.mic_icon.setPixmap(self.mic_on)
#         QApplication.processEvents()

#         # run the heavy pipeline off the UI thread
#         threading.Thread(
#             target=self._record_and_predict_worker, daemon=True
#         ).start()

#     # def _clean_transcript(self, s: str | None) -> str | None:
#     #     if not s:
#     #         return None
#     #     t = s.strip()
#     #     if not t:
#     #         return None

#     #     # Only punctuation / symbols → ignore
#     #     if all(ch in {'.', ',', ' ', '!', '?', '-', '…', '"', "'"} for ch in t):
#     #         return None

#     #     # Must contain at least one letter
#     #     if not any(ch.isalpha() for ch in t):
#     #         return None

#     #     # That's it – allow even very short words like "yes", "sad", "hi"
#     #     return t
#     def _clean_transcript(self, s: str | None) -> str | None:
#         if not s:
#             return None
#         t = s.strip()
#         if not t:
#             return None

#         # 1. Filter common Whisper hallucinations (lowercase)
#         hallucinations = [
#             "thank you", "thanks", "you", "i", "subtitles", 
#             "subtitle by", "copyright", "bye", "watching"
#         ]
#         if t.lower() in hallucinations:
#             print(f"[Filter] Removed hallucination: '{t}'")
#             return None

#         # 2. Only punctuation / symbols → ignore
#         if all(ch in {'.', ',', ' ', '!', '?', '-', '…', '"', "'"} for ch in t):
#             return None

#         # 3. Must contain at least one letter
#         if not any(ch.isalpha() for ch in t):
#             return None
            
#         # 4. Filter single-character inputs (usually noise) unless it's "I" (which we caught above anyway)
#         if len(t) < 2: 
#             return None

#         return t


#     def _record_and_predict_worker(self):
#         try:
#             # --- record (short and bounded) ---
#             sr = FEATURE_SETTINGS.get("sample_rate", 16000)
#             dur = float(self.calib.get("record_seconds", 3.0))
#             use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
#             print(f"[Audio] use_pepper_mic={use_pepper}")

#             if use_pepper:
#                 try:
#                     raw = self.pepper.record(
#                         seconds=int(max(1, round(dur))),
#                         mode=PEPPER.get("record_mode", "seconds"),
#                     )
#                     y, sr_file = _bytes_to_audio(
#                         raw, sr_hint=int(PEPPER.get("sample_rate", 16000))
#                     )
#                     if sr_file != sr:
#                         y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
#                     print("[Audio] source: Pepper mic")
#                 except Exception as e:
#                     traceback.print_exc()
#                     print("[Audio] Pepper record failed, falling back to PC mic:", e)
#                     y = sd.rec(
#                         int(dur * sr), samplerate=sr, channels=1, dtype="float32"
#                     )
#                     sd.wait()
#                     y = y.flatten()
#                     print("[Audio] source: PC mic")
#             else:
#                 y = sd.rec(
#                     int(dur * sr), samplerate=sr, channels=1, dtype="float32"
#                 )
#                 sd.wait()
#                 y = y.flatten()
#                 print("[Audio] source: PC mic")

#             # Speech gate
#             if not self._has_speech(y, sr):
#                 self._add_msg_safe("🎤 (no speech)", is_user=True)
#                 msg = "I couldn't hear any speech. Please try again a bit closer."
#                 self._add_msg_safe(msg, is_user=False)
#                 self._say(msg)
#                 return

#             # Audibility / amplitude guard
#             if float(np.max(np.abs(y))) < float(self.calib.get("min_amp", 0.10)):
#                 reply = RESPONSES.get(
#                     "Uncertain",
#                     "I couldn't hear clearly. Please speak a bit closer.",
#                 )
#                 self._add_msg_safe(reply, is_user=False)
#                 self._say(reply)
#                 return

#             self._add_msg_safe("🎤 (audio captured)", is_user=True)

#             # --- 1) ASR first + early return if no real text ---
#             raw_txt = None
#             transcript = None
#             try:
#                 raw_txt = self.asr.transcribe(y, sr)
#                 print(f"[ASR raw] {raw_txt!r}") 
#                 transcript = self._clean_transcript(raw_txt)
#                 print(f"[ASR clean] {transcript!r}")
#                 # Optional but helpful:
#                 if transcript:
#                     tokens = transcript.lower().split()
#                     if len(tokens) <= 2:
#                         print("[ASR] too short transcript -> treating as no transcript")
#                         transcript = None
#             except Exception:
#                 traceback.print_exc()
#                 transcript = None
#             finally:
#                 self._add_msg_safe(
#                     transcript if transcript else "🎤 (no transcript)",
#                     is_user=True,
#                 )

#             if not transcript:
#                 reply = RESPONSES.get(
#                     "Uncertain",
#                     "I couldn't quite catch any words. Let's try again a bit closer to the mic.",
#                 )
#                 self._add_msg_safe(reply, is_user=False)
#                 self._say(reply)
#                 return

#             # --- 2) SER features + inference (only once, now that we have text) ---
#             feats = (
#                 self._feat_ssl(y, sr)
#                 if self.model_type == "ssl"
#                 else self._feat_mfcc(y, sr)
#             )
#             x = feats[np.newaxis, :, :].astype("float32")
#             logits = self.session.run(None, {self.input_name: x})[0][0]
#             T = max(1e-6, float(self.temperature))
#             probs = softmax(logits / T)
#             label = self._decode(probs)

#             # 3) Lock / override from text
#             if self._should_decay_lock() and label in ("happy", "sad"):
#                 self._lock_emotion(label)

#             self._maybe_override_from_text(transcript)
#             emotion_for_llm = self.emotion_locked or (
#                 label if label in ("happy", "sad") else "unknown"
#             )

#             # 4) Intent detection + LLM
#             intent_direct = False
#             t_low = transcript.lower()
#             for c in [
#                 "how can i",
#                 "how do i",
#                 "what should i",
#                 "tips",
#                 "suggest",
#                 "overcome",
#                 "propose",
#             ]:
#                 if c in t_low:
#                     intent_direct = True
#                     break

#             if intent_direct:
#                 reply = self.chat.reply(emotion_for_llm, transcript)
#                 self.dialog_phase = "chat"
#             else:
#                 if self.dialog_phase == "opener":
#                     reply = RESPONSES.get(
#                         emotion_for_llm,
#                         RESPONSES.get(
#                             "Uncertain",
#                             "I am not sure how you are feeling. Would you like to try again.",
#                         ),
#                     )
#                     self.dialog_phase = "chat"
#                 else:
#                     reply = self.chat.reply(emotion_for_llm, transcript)

#             self.turns_since_lock += 1
#             self._add_msg_safe(reply, is_user=False)

#             debug_path = "debug_pepper.wav"
#             try:
#                 sf.write(debug_path, y, sr, subtype="PCM_16")
#                 print(f"[DEBUG] wrote {debug_path}, shape={y.shape}, sr={sr}")
#             except Exception as e:
#                 print("[DEBUG] failed to write debug_pepper.wav:", e)

#             self.on_prediction(label)
#             self._say(reply)

#         except KeyboardInterrupt:
#             self._add_msg_safe("⏹️ cancelled.", is_user=False)
#             self._finish()
#         except Exception:
#             traceback.print_exc()
#             self._add_msg_safe("Something went wrong. Let's try again.", is_user=False)
#             self._finish()

#     def reset_session(self):
#         self.emotion_locked = None
#         self.emotion_locked_at = None
#         self.turns_since_lock = 0
#         self.chat.history = []
#         self.dialog_phase = "opener"
#         self._add_msg_safe(" New session started.", is_user=False)


# # ---- Run ----
# if __name__ == "__main__":
#     print("[BOOT] OLLAMA_HOST =", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
#     print("[BOOT] OLLAMA_MODEL =", os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
#     app = QApplication(sys.argv)
#     w = EmotionApp()
#     w.show()
#     sys.exit(app.exec_())

#final 0
# # gui_live_predict.py
# import sys, os, json, threading, traceback, tempfile, time
# import numpy as np
# import librosa, sounddevice as sd, pyttsx3
# import soundfile as sf
# import requests
# import onnxruntime as ort
# from io import BytesIO
# from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
# from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QPushButton, QLabel,
#     QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
# )
# from extract_features import extract_mfcc

# # --- Project modules ---
# from config import FEATURE_SETTINGS, RESPONSES, PEPPER
# try:
#     from config import ASSISTANT_STYLE
# except Exception:
#     ASSISTANT_STYLE = (
#         "You are a friendly, everyday wellbeing companion for mild support. "
#         "Keep replies warm, natural, and brief (1–2 short sentences). No diagnosis or clinical terms."
#     )

# try:
#     from pepper_client import PepperClient
# except Exception:
#     PepperClient = None

# # ==== Runtime tweaks (Windows OpenMP noise) ====
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# # ---------------- Files & loading ----------------
# MODELS_DIR = "models"
# TRACKS = [
#     {"name": "SSL v1",  "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"],                           "type": "ssl"},
#     {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], "type": "mfcc"},
# ]

# # ---------------- Utils ----------------
# def softmax(x):
#     x = x - np.max(x)
#     e = np.exp(x)
#     return e / e.sum()


# def _speak_async(text, on_done):
#     def run():
#         try:
#             eng = pyttsx3.init()
#             eng.say(text)
#             eng.runAndWait()
#         except Exception:
#             traceback.print_exc()
#         finally:
#             QTimer.singleShot(0, on_done)
#     threading.Thread(target=run, daemon=True).start()


# def _read_json(path, default):
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return default


# def _find_existing(paths):
#     for p in paths:
#         if os.path.exists(p):
#             return p
#     return None


# def _bytes_to_audio(raw: bytes, sr_hint: int = 16000):
#     """Return (y, sr) where y is float32 mono in [-1, 1]. Handles WAV or raw PCM16."""
#     # WAV/AIFF?
#     if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
#         y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
#         if y.ndim > 1:
#             y = y[:, 0]
#         return y, sr

#     # Otherwise assume little-endian PCM16 mono
#     a = np.frombuffer(raw, dtype=np.int16)
#     if a.size == 0:
#         raise ValueError("Pepper returned empty audio payload")
#     y = (a.astype(np.float32)) / 32768.0
#     return y, sr_hint


# # class LocalASR:
# #     def __init__(self, model_size="small.en"):
# #         from faster_whisper import WhisperModel
# #         # CPU + int8 is fine
# #         self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

# #     def transcribe(self, y, sr):
# #         y = np.asarray(y, dtype=np.float32).reshape(-1)
# #         if y.size == 0: return None

# #         if sr != 16000:
# #             # print(f"[ASR] resampling {sr} -> 16000") 
# #             y = librosa.resample(y, orig_sr=sr, target_sr=16000)
# #             sr = 16000

# #         peak = float(np.max(np.abs(y)))
        
# #         # --- FIX 1: Front Padding ---
# #         # Even with 4 seconds, this guarantees the first word is never lost.
# #         pad_start = np.zeros(int(1.0 * sr), dtype=np.float32)
# #         pad_end = np.zeros(int(0.5 * sr), dtype=np.float32)
# #         y = np.concatenate([pad_start, y, pad_end], axis=0)

# #         # Normalize
# #         if peak > 0: y = (y / peak) * 0.9

# #         fd, path = tempfile.mkstemp(suffix=".wav")
# #         os.close(fd)
# #         try:
# #             sf.write(path, y, sr, subtype="PCM_16")

# #             # --- FIX 2: Conversational Anchors ---
# #             # This teaches Whisper to distinguish "Do you" from "I don't"
# #             prompt_str = (
# #                 "Do you have, I am feeling, Can you help, "
# #                 "sad, sadness, happy, happiness, feeling, emotions, "
# #                 "depression, anxiety, advice, overcome, joy, "
# #                 "what should I do, solution, mental health."
# #             )

# #             segments, info = self.model.transcribe(
# #                 path,
# #                 language="en",
# #                 # --- FIX 3: VAD OFF + Loop Guard ---
# #                 # Safest method for short robotic audio
# #                 vad_filter=False,  
# #                 initial_prompt=prompt_str,
# #                 beam_size=5,
# #             )

# #             texts = []
# #             for s in segments:
# #                 # Loop Guard: Stop if text goes beyond audio length
# #                 if s.start > 6.0: 
# #                     break 
# #                 texts.append(s.text)

# #             text = " ".join(texts).strip()
# #             print(f"[ASR raw text] {text!r}")
# #             return text

# #         except Exception as e:
# #             print("[ASR ERROR]", repr(e))
# #             return None
# #         finally:
# #             try: os.remove(path)
# #             except: pass

# class LocalASR:
#     def __init__(self, model_size="small.en"):
#         from faster_whisper import WhisperModel
#         # CPU + int8 is fine
#         self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

#     def transcribe(self, y, sr):
#         y = np.asarray(y, dtype=np.float32).reshape(-1)
#         if y.size == 0:
#             return None

#         # Always run Whisper at 16 kHz
#         if sr != 16000:
#             y = librosa.resample(y, orig_sr=sr, target_sr=16000)
#             sr = 16000

#         peak = float(np.max(np.abs(y)))
#         duration = len(y) / sr
#         print(f"[ASR] dur={duration:.2f}s, sr={sr}, peak={peak:.4f}")

#         # Light normalize
#         if peak > 0:
#             y = (y / peak) * 0.9

#         # Small padding so first/last word is not cut
#         pad = np.zeros(int(0.5 * sr), dtype=np.float32)
#         y = np.concatenate([pad, y, pad], axis=0)

#         fd, path = tempfile.mkstemp(suffix=".wav")
#         os.close(fd)
#         try:
#             sf.write(path, y, sr, subtype="PCM_16")

#             # IMPORTANT: only high-level emotional words, no full questions
#             prompt_str = (
#                 "emotion, feelings, sad, sadness, happy, happiness, "
#                 "lonely, tired, worried, anxious, calm, relaxed, support, help."
#             )

#             segments, info = self.model.transcribe(
#                 path,
#                 language="en",
#                 vad_filter=True,      # <--- TURN VAD BACK ON
#                 initial_prompt=prompt_str,
#                 beam_size=5,
#             )

#             texts = []
#             last_norm = None
#             total_speech = 0.0

#             for s in segments:
#                 seg_text = s.text.strip()
#                 if not seg_text:
#                     continue
#                 total_speech += float(s.end - s.start)

#                 # skip exact duplicate segments
#                 norm = seg_text.lower()
#                 if norm == last_norm:
#                     continue
#                 last_norm = norm

#                 print(f"[ASR seg] {s.start:.2f}-{s.end:.2f}s: {seg_text!r}")
#                 texts.append(seg_text)

#             # If Whisper found almost no speech, treat as no transcript
#             if total_speech < 0.4:
#                 print(f"[ASR] very little speech detected ({total_speech:.2f}s) -> None")
#                 return None

#             text = " ".join(texts).strip()
#             if not text:
#                 print("[ASR] empty transcript from Whisper")
#                 return None

#             print(f"[ASR raw text] {text!r}")
#             return text

#         except Exception as e:
#             print("[ASR ERROR]", repr(e))
#             traceback.print_exc()
#             return None
#         finally:
#             try:
#                 os.remove(path)
#             except Exception:
#                 pass


# class ChatEngine:
#     def __init__(self, model=None, host=None, debug=True):
#         self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#         self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
#         self.url_chat = f"{self.host}/api/chat"
#         self.url_gen = f"{self.host}/api/generate"
#         self.system = ASSISTANT_STYLE
#         self.history = []
#         self.debug = debug
#         try:
#             ok = requests.get(f"{self.host}/api/tags", timeout=5).ok
#             print(f"[Ollama] host={self.host} online={ok}")
#         except Exception as e:
#             print(f"[Ollama] ping failed: {e}")

#     def _make_prompt(self, emotion: str, transcript: str | None) -> str:
#         t = (transcript or "").strip()
        
#         # We tell the LLM explicitly how to interpret the data
#         return (
#             f"{self.system}\n"
#             f"--- Context Info ---\n"
#             f"DETECTED EMOTION FROM VOICE TONE: {emotion.upper()}\n"
#             f"TRANSCRIPT OF SPEECH: {t if t else '(empty)'}\n"
#             f"--------------------\n"
#             "IMPORTANT: The transcript might have errors due to accent or noise.\n"
#             f"If the transcript says 'finished' or 'set' but the Emotion is SAD, interpret it as 'feeling sad' or 'upset'.\n"
#             "Respond naturally to the user's emotion and words."
#         )

#     def _compose_text_prompt(self, prompt: str) -> str:
#         parts = [f"System: {self.system}"]
#         for m in self.history[-8:]:
#             parts.append(f"{m['role'].capitalize()}: {m['content']}")
#         parts.append(f"User: {prompt}")
#         parts.append("Assistant:")
#         return "\n".join(parts)

#     def reply(self, emotion: str, transcript: str | None) -> str:
#         prompt = self._make_prompt(emotion, transcript)

#         try:
#             text_prompt = self._compose_text_prompt(prompt)
#             print(f"[Ollama] POST {self.url_gen} model={self.model}")
#             r = requests.post(
#                 self.url_gen,
#                 json={
#                     "model": self.model,
#                     "prompt": text_prompt,
#                     "stream": False,
#                     "options": {
#                         "temperature": 0.6,
#                         "repeat_penalty": 1.15,
#                         "num_predict": 60,
#                     },
#                 },
#                 # --- FIX 4: 60s Timeout ---
#                 timeout=60,
#             )
#             r.raise_for_status()
#             text = (r.json().get("response") or "").strip()
#             if not text:
#                 raise RuntimeError("empty generate response")

#             self.history += [
#                 {"role": "user", "content": prompt},
#                 {"role": "assistant", "content": text},
#             ]
#             self.history = self.history[-12:]
#             return text

#         except Exception as e:
#             if self.debug:
#                 print("[Ollama generate error]", repr(e))
#             return (
#                 "I’m having a hiccup reaching my language model. "
#                 "We can keep talking, or try again shortly."
#             )


# # ---------------- Chat UI ----------------
# class ChatBubble(QLabel):
#     def __init__(self, text, is_user=False):
#         super().__init__(text)
#         self.setWordWrap(True)
#         self.setMaximumWidth(360)
#         color = "#e0e0e0" if is_user else "#95abbe"
#         self.setStyleSheet(f"background:{color}; border-radius:10px; padding:8px;")


# # ---------------- Main App ----------------
# class EmotionApp(QWidget):
#     sig_add_msg = pyqtSignal(str, bool)  # text, is_user
#     sig_finish = pyqtSignal()
#     sig_update_emoji = pyqtSignal(str)    #for emoji


#     def __init__(self):
#         super().__init__()
#         self._last_click = 0.0
#         self.setWindowTitle("Speech Emotion Detection Application")
#         self.setGeometry(100, 100, 400, 500)
#         self.emotion_locked = None
#         self.emotion_locked_at = None
#         self.turns_since_lock = 0
#         self.sig_update_emoji.connect(self.update_emoji) 
        
#         # top bar
#         self.mic_icon = QLabel()
#         self.mic_off = QPixmap("mic-off.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_on = QPixmap("mic-on.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_icon.setPixmap(self.mic_off)

#         # --- EMOJI FEEDBACK LABEL ---
#         self.emoji_label = QLabel("😐") # Default neutral
#         self.emoji_label.setAlignment(Qt.AlignCenter)
#         self.emoji_label.setStyleSheet("font-size: 72px; color: #888888;")

#         # Chat area
#         self.chat_layout = QVBoxLayout()
#         self.chat_layout.setAlignment(Qt.AlignTop)
#         container = QWidget()
#         container.setLayout(self.chat_layout)
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)
#         scroll.setWidget(container)
#         self.scroll = scroll

#         # Record button (single control)
#         self.button = QPushButton("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setIconSize(QSize(20, 20))
#         self.button.clicked.connect(self.record_and_predict)

#         # Layout
#         main = QVBoxLayout(self)
#         main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
#         main.addWidget(self.emoji_label, alignment=Qt.AlignCenter)
#         main.addWidget(scroll)
#         main.addWidget(self.button)

#         # Runtime state
#         self.is_processing = False
#         self.session = None
#         self.input_name = None
#         self.model_type = None
#         self.classes = ["happy", "sad"]
#         self.calib = {
#             "mode": "threshold",
#             "sad_threshold": 0.57,
#             "min_confidence": 0.50,
#             "min_amp": 0.10,
#             "record_seconds": 4.0,  # <--- SET TO 4.0s (Safety + Speed Balance)
#         }
#         self.temperature = 1.0
#         self.ssl = None

#         # Conversation engines/state
#         self.asr = LocalASR()
#         self.chat = ChatEngine(debug=True)
#         self.dialog_phase = "opener"
#         self.sig_add_msg.connect(self._add_msg)
#         self.sig_finish.connect(self._finish_ui)

#         self._auto_load_model()
#         self.pepper = None
#         if PEPPER.get("enabled") and PepperClient:
#             try:
#                 self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
#                 self.pepper.connect()
#                 print("[Pepper] connected")
#             except Exception as e:
#                 print("[Pepper] connect failed:", e)
#                 self.pepper = None

#     def update_emoji(self, emotion):
#         if emotion == "happy":
#             self.emoji_label.setText("😃")
#             self.emoji_label.setStyleSheet("font-size: 72px; color: #2ecc71;") 
#         elif emotion == "sad":
#             self.emoji_label.setText("😔")
#             self.emoji_label.setStyleSheet("font-size: 72px; color: #e74c3c;") 
#         elif emotion == "listening":
#             self.emoji_label.setText("🙂")
#             self.emoji_label.setStyleSheet("font-size: 72px; color: #3498db;") 
#         elif emotion == "thinking":
#             self.emoji_label.setText("🤔")
#             self.emoji_label.setStyleSheet("font-size: 72px; color: #f1c40f;")
#         else:
#             self.emoji_label.setText("😐")
#             self.emoji_label.setStyleSheet("font-size: 72px; color: #888888;") 

#     # --- Speech gate ---
#     def _has_speech(self, y, sr):
#         if y is None: return False
#         y = np.asarray(y, dtype=np.float32)
#         if y.size == 0: return False

#         drop = int(0.25 * sr)
#         if y.size > drop:
#             y = y[drop:]

#         peak = float(np.max(np.abs(y)))
#         rms  = float(np.sqrt(np.mean(np.square(y))))
#         frame = max(1, int(0.030 * sr))
#         hop   = max(1, int(0.015 * sr))
#         zcr   = float(librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop).mean())
#         p95 = float(np.percentile(np.abs(y), 95))
#         loud_ratio = float(np.mean(np.abs(y) > 0.02))

#         # --- LOW SENSITIVITY THRESHOLDS ---
#         peak_th = 0.04      
#         rms_th  = 0.008     
#         p95_th  = 0.020     
#         loud_th = 0.050     
#         zcr_th  = 0.015     

#         passed = (
#             peak >= peak_th
#             and rms  >= rms_th
#             and p95  >= p95_th
#             and zcr  >= zcr_th
#             and loud_ratio >= loud_th
#         )

#         print(
#             f"[Gate] peak={peak:.4f} rms={rms:.4f} zcr={zcr:.4f} "
#             f"p95={p95:.4f} loud={loud_ratio:.3f} -> {'PASS' if passed else 'BLOCK'}"
#         )
#         return passed


#     def _should_decay_lock(self, minutes=7, max_turns=4):
#         if self.emotion_locked is None:
#             return True
#         if self.emotion_locked_at and (time.time() - self.emotion_locked_at) > minutes * 60:
#             return True
#         if self.turns_since_lock >= max_turns:
#             return True
#         return False

#     def _lock_emotion(self, label):
#         self.emotion_locked = label
#         self.emotion_locked_at = time.time()
#         self.turns_since_lock = 0

#     def _say(self, text):
#         def finish_on_ui():
#             QTimer.singleShot(0, self._finish)

#         if self.pepper:
#             def run():
#                 try:
#                     self.pepper.tts(text)
#                 except Exception:
#                     traceback.print_exc()
#                 finally:
#                     finish_on_ui()
#             threading.Thread(target=run, daemon=True).start()
#         else:
#             _speak_async(text, finish_on_ui)

#     def _maybe_override_from_text(self, transcript: str | None):
#         if not transcript:
#             return
#         t = transcript.lower()
#         if any(p in t for p in ["i'm happy", "i am happy", "feeling happy"]):
#             self._lock_emotion("happy")
#         elif any(p in t for p in ["i'm sad", "i am sad", "feeling sad", "feeling down", "upset"]):
#             self._lock_emotion("sad")

#     def on_prediction(self, label: str):
#         pass

#     def _add_msg(self, text, is_user=False):
#         bubble = ChatBubble(text, is_user)
#         row = QHBoxLayout()
#         if is_user:
#             row.addStretch()
#             row.addWidget(bubble)
#         else:
#             row.addWidget(bubble)
#             row.addStretch()
#         f = QFrame()
#         f.setLayout(row)
#         self.chat_layout.addWidget(f)
#         sb = self.scroll.verticalScrollBar()
#         sb.setValue(sb.maximum())

#     def _add_msg_safe(self, text, is_user=False):
#         self.sig_add_msg.emit(text, is_user)

#     def _auto_load_model(self):
#         try:
#             chosen = None
#             for t in TRACKS:
#                 tdir = t["dir"]
#                 onnx_path = _find_existing([os.path.join(tdir, fn) for fn in t["onnx"]])
#                 if onnx_path:
#                     chosen = (t, onnx_path)
#                     break
#             if not chosen:
#                 raise FileNotFoundError("No ONNX model found in models/ssl_v1 or models/mfcc_v1.")

#             track, onnx_path = chosen
#             self.model_type = track["type"]
#             print(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")

#             calib_path = os.path.join(track["dir"], "calibration.json")
#             self.calib.update(_read_json(calib_path, {}))
#             temp = _read_json(os.path.join(track["dir"], "temperature.json"), {"temperature": 1.0})
#             self.temperature = float(temp.get("temperature", 1.0))
#             self.classes = _read_json(calib_path, {}).get("classes", self.classes)

#             so = ort.SessionOptions()
#             so.intra_op_num_threads = 1
#             so.inter_op_num_threads = 1
#             so.log_severity_level = 3
#             self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
#             self.input_name = self.session.get_inputs()[0].name

#             if self.model_type == "ssl" and self.ssl is None:
#                 import warnings
#                 try:
#                     from transformers.utils import logging as hf_logging
#                     hf_logging.set_verbosity_error()
#                 except Exception: pass
#                 warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing`.*")
#                 warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces`.*")
#                 from ssl_frontend import SSLFrontend
#                 self.ssl = SSLFrontend()

#         except Exception:
#             traceback.print_exc()
#             self._add_msg_safe("Setup problem. Please check model files.", is_user=False)

#     def _feat_mfcc(self, y, sr):
#         target = FEATURE_SETTINGS.get("sample_rate", 16000)
#         if sr != target:
#             y = librosa.resample(y, orig_sr=sr, target_sr=target)
#             sr = target
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr):
#                 y = yt
#         except Exception: pass
#         return extract_mfcc(array=y, sr=sr)

#     def _feat_ssl(self, y, sr):
#         if sr != 16000:
#             y = librosa.resample(y, orig_sr=sr, target_sr=16000)
#             sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr):
#                 y = yt
#         except Exception: pass
#         return self.ssl(y)

#     def _decode(self, probs):
#         try:
#             idx_h = self.classes.index("happy")
#             idx_s = self.classes.index("sad")
#         except ValueError:
#             idx_h, idx_s = 0, 1
#         p_h, p_s = float(probs[idx_h]), float(probs[idx_s])
#         p_max = max(p_h, p_s)
#         if p_max < float(self.calib.get("min_confidence", 0.50)):
#             return "Uncertain"
#         if self.calib.get("mode", "threshold") == "threshold":
#             return "sad" if p_s >= float(self.calib.get("sad_threshold", 0.57)) else "happy"
#         return "happy" if p_h >= p_s else "sad"

#     def _finish_ui(self):
#         self.mic_icon.setPixmap(self.mic_off)
#         self.button.setText("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setEnabled(True)
#         self.is_processing = False

#     def _finish(self):
#         self.sig_finish.emit()

#     def record_and_predict(self):
#         now = time.monotonic()
#         if now - self._last_click < 0.8:
#             return
#         self._last_click = now
#         if self.is_processing or self.session is None:
#             return
#         self.is_processing = True
#         self.button.setEnabled(False)
#         self.button.setText("🔴  Listening…")
#         self.button.setIcon(QIcon("mic-on.png"))
#         self.mic_icon.setPixmap(self.mic_on)

#         # --- NEW: Update Emoji to Listening ---
#         self.update_emoji("listening")

#         QApplication.processEvents()
#         threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

#     def _clean_transcript(self, s: str | None) -> str | None:
#         if not s: return None
#         t = s.strip()
#         t_low = t.lower()
#         if not t: return None

#         # --- 1. CATCH LOOPING HALLUCINATIONS (The New Fix) ---
#         # Whisper often loops the prompt phrases on silence.
#         # If we see the same phrase repeated, we delete it.
        
#         # Check for "What should I do" loop
#         if t_low.count("what should i do") >= 2:
#             print(f"[Filter] Removed looping prompt: '{t}'")
#             return None
             
#         # Check for "I am feeling" loop
#         if t_low.count("i am feeling") >= 2:
#             print(f"[Filter] Removed looping prompt: '{t}'")
#             return None

#         # --- 2. Technical Garbage ---
#         technical_garbage = [
#             "subtitle by", "subtitles", "copyright", "amara.org", 
#             "community", "contributed by", "transcribed by"
#         ]
#         for garbage in technical_garbage:
#             if garbage in t_low:
#                 print(f"[Filter] Removed technical garbage: '{t}'")
#                 return None

#         # --- 3. Social/Short Hallucinations ---
#         # If it's just "Thank you" or "You" without other words, assume it's fake

#         if len(t) < 5 and t_low not in ["hi", "no", "yes", "ok", "hey", "bye"]:
#             print(f"[Filter] Too few words -> '{t}'")
#             return None

#         # --- 4. Basic Cleaning ---
#         # Must contain at least one letter
#         if not any(ch.isalpha() for ch in t):
#             return None

#         return t


#     def _record_and_predict_worker(self):
#         try:
#             # --- record (short and bounded) ---
#             sr = FEATURE_SETTINGS.get("sample_rate", 16000)
#             dur = float(self.calib.get("record_seconds", 4.0)) # Uses 4.0 from init
#             use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
#             print(f"[Audio] use_pepper_mic={use_pepper}")

#             if use_pepper:
#                 try:
#                     raw = self.pepper.record(
#                         seconds=int(max(1, round(dur))),
#                         mode=PEPPER.get("record_mode", "seconds"),
#                     )
#                     # --- CRITICAL FIX FOR PEPPER 48k ---
#                     y, sr_file = _bytes_to_audio(
#                         raw, sr_hint=48000 
#                     )
#                     if sr_file != sr:
#                         y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
#                     print("[Audio] source: Pepper mic")
#                 except Exception as e:
#                     traceback.print_exc()
#                     print("[Audio] Pepper record failed, falling back to PC mic:", e)
#                     y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
#                     sd.wait()
#                     y = y.flatten()
#                     print("[Audio] source: PC mic")
#             else:
#                 y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
#                 sd.wait()
#                 y = y.flatten()
#                 print("[Audio] source: PC mic")

#             # Speech gate
#             if not self._has_speech(y, sr):
#                 self._add_msg_safe("🎤 (no speech)", is_user=True)
#                 msg = "I couldn't hear any speech. Please try again a bit closer."
#                 self._add_msg_safe(msg, is_user=False)
#                 self.update_emoji("neutral") # Reset emoji
#                 self._say(msg)
#                 return

#             # Audibility / amplitude guard
#             if float(np.max(np.abs(y))) < float(self.calib.get("min_amp", 0.10)):
#                 reply = RESPONSES.get(
#                     "Uncertain",
#                     "I couldn't hear clearly. Please speak a bit closer.",
#                 )
#                 self._add_msg_safe(reply, is_user=False)
#                 self.update_emoji("neutral") # Reset emoji
#                 self._say(reply)
#                 return

#             self._add_msg_safe("🎤 (audio captured)", is_user=True)

#             # --- Switch Emoji to Thinking (Hides Latency) ---
#             self.sig_update_emoji.emit("thinking")
#             # -----------------------------------------------

#             # --- 1) ASR first + early return if no real text ---
#             raw_txt = None
#             transcript = None
#             try:
#                 raw_txt = self.asr.transcribe(y, sr)
#                 transcript = self._clean_transcript(raw_txt)
#                 if transcript and transcript.lower().strip(" .!") in ["thank you", "thanks", "you", "bye"]:
#                     # Check the peak volume we measured earlier
#                     current_peak = float(np.max(np.abs(y)))
#                     # If it's a "Thank you" but very quiet (< 0.15), it's probably a hallucination
#                     if current_peak < 0.15: 
#                         print(f"[Smart Filter] Rejected quiet hallucination: '{transcript}' (Peak: {current_peak:.3f})")
#                         transcript = None
#             except Exception:
#                 traceback.print_exc()
#                 transcript = None
#             finally:
#                 self._add_msg_safe(
#                     transcript if transcript else " (no transcript)",
#                     is_user=True,
#                 )

#             if not transcript:
#                 reply = RESPONSES.get(
#                     "Uncertain",
#                     "I couldn't quite catch any words. Let's try again.",
#                 )
#                 self._add_msg_safe(reply, is_user=False)
#                 self.sig_update_emoji.emit("neutral") # Reset emoji
#                 self._say(reply)
#                 return

#             # --- 2) SER features + inference (only once, now that we have text) ---
#             feats = (
#                 self._feat_ssl(y, sr)
#                 if self.model_type == "ssl"
#                 else self._feat_mfcc(y, sr)
#             )
#             x = feats[np.newaxis, :, :].astype("float32")
#             logits = self.session.run(None, {self.input_name: x})[0][0]
#             T = max(1e-6, float(self.temperature))
#             probs = softmax(logits / T)
#             label = self._decode(probs)

#             # 3) Lock / override from text
#             if self._should_decay_lock() and label in ("happy", "sad"):
#                 self._lock_emotion(label)
            
#             # --- NEW: Update the UI with the detected label ---
#             # Use 'happy' or 'sad' from the prediction
#             self.sig_update_emoji.emit(label)

#             self._maybe_override_from_text(transcript)

#             emotion_for_llm = self.emotion_locked or (
#                 label if label in ("happy", "sad") else "unknown"
#             )

#             # 4) Intent detection
#             intent_direct = False
#             t_low = transcript.lower()
#             for c in ["how can i", "how do i", "what should i", "tips", "suggest", "overcome", "solution"]:
#                 if c in t_low:
#                     intent_direct = True
#                     break

#             # --- CORRECTED LOGIC FLOW ---
            
#             # 1. ALWAYS check the Opener first (Did we greet them yet?)
#             if self.dialog_phase == "opener":
#                 reply = RESPONSES.get(
#                     emotion_for_llm,
#                     RESPONSES.get("Uncertain", "I am not sure, but I am here for you.")
#                 )
#                 self.dialog_phase = "chat"
            
#             # 2. THEN check if they asked a specific question
#             elif intent_direct:
#                 reply = self.chat.reply(emotion_for_llm, transcript)
#                 self.dialog_phase = "chat"
                
#             # 3. Otherwise, just chat normally
#             else:
#                 reply = self.chat.reply(emotion_for_llm, transcript)

#             self.turns_since_lock += 1
#             self._add_msg_safe(reply, is_user=False)

#             # Debug save
#             try:
#                 sf.write("debug_pepper.wav", y, sr, subtype="PCM_16")
#             except Exception: pass

#             self.on_prediction(label)
#             self._say(reply)

#         except KeyboardInterrupt:
#             self._add_msg_safe("cancelled.", is_user=False)
#             self._finish()
#         except Exception:
#             traceback.print_exc()
#             self._add_msg_safe("Something went wrong. Let's try again.", is_user=False)
#             self._finish()

#     def reset_session(self):
#         self.emotion_locked = None
#         self.emotion_locked_at = None
#         self.turns_since_lock = 0
#         self.chat.history = []
#         self.dialog_phase = "opener"
#         self._add_msg_safe("New session started.", is_user=False)


# # ---- Run ----
# if __name__ == "__main__":
#     print("[BOOT] OLLAMA_HOST =", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
#     app = QApplication(sys.argv)
#     w = EmotionApp()
#     w.show()
#     sys.exit(app.exec_())

# --------- Working code ----------------""
# import sys, os, json, threading, traceback, tempfile, time
# import numpy as np
# import librosa, sounddevice as sd, pyttsx3
# import soundfile as sf
# import requests
# import onnxruntime as ort
# from io import BytesIO
# from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
# from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QPushButton, QLabel,
#     QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
# )
# from extract_features import extract_mfcc

# # --- Project modules ---
# from config import FEATURE_SETTINGS, RESPONSES, PEPPER
# try:
#     from config import ASSISTANT_STYLE
# except Exception:
#     ASSISTANT_STYLE = (
#         "You are a friendly, everyday wellbeing companion for mild support. "
#         "Keep replies warm, natural, and brief (1–2 short sentences). No diagnosis or clinical terms."
#     )

# try:
#     from pepper_client import PepperClient
# except Exception:
#     PepperClient = None

# # ==== Runtime tweaks (Windows OpenMP noise) ====
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# # ---------------- Files & loading ----------------
# MODELS_DIR = "models"
# TRACKS = [
#     {"name": "SSL v1",  "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"],                           "type": "ssl"},
#     {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], "type": "mfcc"},
# ]

# # ---------------- Utils ----------------
# def softmax(x):
#     x = x - np.max(x); e = np.exp(x); return e / e.sum()

# def _speak_async(text, on_done):
#     def run():
#         try:
#             eng = pyttsx3.init()
#             eng.say(text); eng.runAndWait()
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

# # --- LOCAL ASR (Using your robust "Old Code" settings) ---
# class LocalASR:
#     def __init__(self, model_size="small"): 
#         from faster_whisper import WhisperModel
#         self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

#     def transcribe(self, y, sr):
#         y = np.asarray(y, dtype=np.float32).reshape(-1)
#         if y.size == 0: return None

#         # Normalization & Padding
#         peak = float(np.max(np.abs(y)))
#         if peak > 0: y = (y / peak) * 0.9
#         pad = np.zeros(int(0.25 * sr), dtype=np.float32)
#         y = np.concatenate([pad, y, pad], axis=0)

#         fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
#         try:
#             sf.write(path, y, sr, subtype="PCM_16")
#             # vad_filter=False is SAFER for short clips (prevents cutoff)
#             segments, _ = self.model.transcribe(
#                 path, language="en", vad_filter=False, beam_size=1
#             )
#             text = " ".join(s.text for s in segments).strip()
#             return text or None
#         finally:
#             try: os.remove(path)
#             except Exception: pass

# class ChatEngine:
#     def __init__(self, model=None, host=None, debug=True):
#         self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#         self.host  = (host  or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
#         self.url_chat = f"{self.host}/api/chat"
#         self.url_gen  = f"{self.host}/api/generate"
#         self.system = ASSISTANT_STYLE
#         self.history = []
#         self.debug = debug
#         try:
#             requests.get(f"{self.host}/api/tags", timeout=5)
#         except Exception as e:
#             print(f"[Ollama] ping failed: {e}")

#     def _make_prompt(self, emotion: str, transcript: str | None) -> str:
#         t = (transcript or "").strip()
#         return (
#             f"{self.system}\n"
#             f"UserEmotion: {emotion or 'unknown'}\n"
#             f"UserSaid: {t if t else '(empty)'}\n"
#             "If UserSaid looks like a question, answer it directly with 1–2 concrete suggestions. "
#             "Otherwise reply in 1–2 short, natural sentences."
#         )

#     def _compose_text_prompt(self, prompt: str) -> str:
#         parts = [f"System: {self.system}"]
#         for m in self.history[-8:]:
#             parts.append(f"{m['role'].capitalize()}: {m['content']}")
#         parts.append(f"User: {prompt}")
#         parts.append("Assistant:")
#         return "\n".join(parts)

#     def reply(self, emotion: str, transcript: str | None) -> str:
#         prompt = self._make_prompt(emotion, transcript)
#         try:
#             text_prompt = self._compose_text_prompt(prompt)
#             print(f"[Ollama] POST {self.url_gen} model={self.model}")
#             r = requests.post(
#                 self.url_gen,
#                 json={
#                     "model": self.model, "prompt": text_prompt, "stream": False,
#                     "options": {"temperature": 0.6, "repeat_penalty": 1.15, "num_predict": 90},
#                 },
#                 timeout=60, 
#             )
#             r.raise_for_status()
#             text = (r.json().get("response") or "").strip()
#             if not text: raise RuntimeError("empty generate response")
#             self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
#             self.history = self.history[-12:]
#             return text
#         except Exception as e:
#             if self.debug: print("[Ollama error]", repr(e))
#             return "I’m having a hiccup reaching my language model. We can keep talking, or try again shortly."

# # ---------------- Chat UI ----------------
# class ChatBubble(QLabel):
#     def __init__(self, text, is_user=False):
#         super().__init__(text)
#         self.setWordWrap(True)
#         self.setMaximumWidth(360)
#         color = "#e0e0e0" if is_user else "#95abbe"
#         self.setStyleSheet(f"background:{color}; border-radius:10px; padding:8px;")

# # ---------------- Main App ----------------
# class EmotionApp(QWidget):
#     sig_add_msg = pyqtSignal(str, bool)
#     sig_finish  = pyqtSignal()
#     sig_update_emoji = pyqtSignal(str) # Signal for Emojis

#     def __init__(self):
#         super().__init__()
#         self._last_click = 0.0
#         self.setWindowTitle("Speech Emotion Detection Application")
#         self.setGeometry(100, 100, 400, 500)
#         self.emotion_locked = None
#         self.emotion_locked_at = None
#         self.turns_since_lock = 0
#         self.last_gate_stats = None
        
#         self.sig_update_emoji.connect(self.update_emoji)

#         # top bar
#         self.mic_icon = QLabel()
#         self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_icon.setPixmap(self.mic_off)

#         # --- EMOJI UI ---
#         self.emoji_label = QLabel("😐")
#         self.emoji_label.setAlignment(Qt.AlignCenter)
#         self.emoji_label.setStyleSheet("font-size: 72px; color: #888888;")

#         # Chat area
#         self.chat_layout = QVBoxLayout()
#         self.chat_layout.setAlignment(Qt.AlignTop)
#         container = QWidget(); container.setLayout(self.chat_layout)
#         scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container)
#         self.scroll = scroll

#         # Record button
#         self.button = QPushButton("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setIconSize(QSize(20, 20))
#         self.button.clicked.connect(self.record_and_predict)

#         # Layout
#         main = QVBoxLayout(self)
#         main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
#         main.addWidget(self.emoji_label, alignment=Qt.AlignCenter)
#         main.addWidget(scroll)
#         main.addWidget(self.button)

#         # Runtime state
#         self.is_processing = False
#         self.session = None
#         self.input_name = None
#         self.model_type = None
#         self.classes = ["happy", "sad"]
#         # Using your working calibration
#         self.calib = {"mode":"threshold","sad_threshold":0.57,"min_confidence":0.50,"min_amp":0.1,"record_seconds":3.0}
#         self.temperature = 1.0
#         self.ssl = None

#         # Engines
#         self.asr  = LocalASR()
#         self.chat = ChatEngine(debug=True)
#         self.dialog_phase = "opener"
#         self.sig_add_msg.connect(self._add_msg)
#         self.sig_finish.connect(self._finish_ui)

#         self._auto_load_model()
#         self.pepper = None
#         if PEPPER.get("enabled") and PepperClient:
#             try:
#                 self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
#                 self.pepper.connect()
#                 print("[Pepper] connected. Flushing buffer...")
                
#                 # --- BUFFER FLUSH FIX (Prevents 2nd run error) ---
#                 try:
#                     self.pepper.record(seconds=2, mode="seconds")
#                     print("[Pepper] Buffer flushed.")
#                 except Exception: pass
                
#             except Exception as e:
#                 print("[Pepper] connect failed:", e)
#                 self.pepper = None

#     def update_emoji(self, emotion):
#         # Updates the UI thread-safely
#         if emotion == "happy":
#             self.emoji_label.setText("😃"); self.emoji_label.setStyleSheet("font-size: 72px; color: #2ecc71;")
#         elif emotion == "sad":
#             self.emoji_label.setText("😔"); self.emoji_label.setStyleSheet("font-size: 72px; color: #e74c3c;")
#         elif emotion == "listening":
#             self.emoji_label.setText("🙂"); self.emoji_label.setStyleSheet("font-size: 72px; color: #3498db;")
#         elif emotion == "thinking":
#             self.emoji_label.setText("🤔"); self.emoji_label.setStyleSheet("font-size: 72px; color: #f1c40f;")
#         else:
#             self.emoji_label.setText("😐"); self.emoji_label.setStyleSheet("font-size: 72px; color: #888888;")

#     def _has_speech(self, y, sr):
#         if y is None or y.size == 0: return False
#         drop = int(0.25 * sr)
#         if y.size > drop: y = y[drop:]

#         peak = float(np.max(np.abs(y)))
#         rms  = float(np.sqrt(np.mean(np.square(y))))
#         frame = max(1, int(0.030 * sr)); hop = max(1, int(0.015 * sr))
#         zcr   = float(librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop).mean())
#         p95   = float(np.percentile(np.abs(y), 95))

#         # --- Robust Thresholds (from your working code) ---
#         peak_th, rms_th, zcr_th, p95_th = 0.03, 0.015, 0.025, 0.018

#         passed = (peak >= peak_th) and (rms >= rms_th) and (zcr >= zcr_th) and (p95 >= p95_th)
#         print(f"[Gate] rms={rms:.4f} zcr={zcr:.4f} peak={peak:.4f} p95={p95:.4f} -> {'PASS' if passed else 'BLOCK'}")
#         self.last_gate_stats = {"peak": peak, "rms": rms, "zcr": zcr, "p95": p95}
#         return passed

#     def _should_decay_lock(self, minutes=7, max_turns=4):
#         if self.emotion_locked is None: return True
#         if self.emotion_locked_at and (time.time() - self.emotion_locked_at) > minutes * 60: return True
#         if self.turns_since_lock >= max_turns: return True
#         return False

#     def _lock_emotion(self, label):
#         self.emotion_locked = label
#         self.emotion_locked_at = time.time()
#         self.turns_since_lock = 0

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

#     def _maybe_override_from_text(self, transcript: str | None):
#         if not transcript: return
#         t = transcript.lower()
#         if any(p in t for p in ["i'm happy", "i am happy", "feeling happy"]): self._lock_emotion("happy")
#         elif any(p in t for p in ["i'm sad", "i am sad", "feeling sad", "feeling down", "upset"]): self._lock_emotion("sad")

#     def on_prediction(self, label: str): pass

#     def _add_msg(self, text, is_user=False):
#         bubble = ChatBubble(text, is_user)
#         row = QHBoxLayout()
#         if is_user: row.addStretch(); row.addWidget(bubble)
#         else: row.addWidget(bubble); row.addStretch()
#         f = QFrame(); f.setLayout(row); self.chat_layout.addWidget(f)
#         self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

#     def _add_msg_safe(self, text, is_user=False): self.sig_add_msg.emit(text, is_user)

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
#             temp = _read_json(os.path.join(track["dir"], "temperature.json"), {"temperature":1.0})
#             self.temperature = float(temp.get("temperature", 1.0))
#             self.classes = _read_json(calib_path, {}).get("classes", self.classes)

#             so = ort.SessionOptions()
#             so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.log_severity_level = 3
#             self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
#             self.input_name = self.session.get_inputs()[0].name

#             if self.model_type == "ssl" and self.ssl is None:
#                 import warnings; warnings.simplefilter("ignore")
#                 from ssl_frontend import SSLFrontend; self.ssl = SSLFrontend()
#         except Exception:
#             traceback.print_exc()
#             self._add_msg_safe("Setup problem.", is_user=False)

#     def _feat_mfcc(self, y, sr):
#         if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr): y = yt
#         except Exception: pass
#         return extract_mfcc(array=y, sr=sr)

#     def _feat_ssl(self, y, sr):
#         if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr): y = yt
#         except Exception: pass
#         return self.ssl(y)

#     def _decode(self, probs):
#         try: idx_h = self.classes.index("happy"); idx_s = self.classes.index("sad")
#         except ValueError: idx_h, idx_s = 0, 1
#         p_h, p_s = float(probs[idx_h]), float(probs[idx_s])
#         if max(p_h, p_s) < float(self.calib.get("min_confidence", 0.50)): return "Uncertain"
#         if self.calib.get("mode", "threshold") == "threshold":
#             return "sad" if p_s >= float(self.calib.get("sad_threshold", 0.57)) else "happy"
#         return "happy" if p_h >= p_s else "sad"

#     def _finish_ui(self):
#         self.mic_icon.setPixmap(self.mic_off)
#         self.button.setText("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setEnabled(True)
#         self.is_processing = False

#     def _finish(self): self.sig_finish.emit()

#     def record_and_predict(self):
#         now = time.monotonic()
#         if now - self._last_click < 0.8: return
#         self._last_click = now
#         if self.is_processing or self.session is None: return
#         self.is_processing = True
#         self.button.setEnabled(False)
#         self.button.setText("🔴  Listening…")
#         self.button.setIcon(QIcon("mic-on.png"))
#         self.mic_icon.setPixmap(self.mic_on)
        
#         self.sig_update_emoji.emit("listening") # Signal to update Face
#         QApplication.processEvents()
#         threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

#     def _clean_transcript(self, s: str | None) -> str | None:
#         if not s: return None
#         t = s.strip(); t_low = t.lower()
#         if not t: return None

#         # --- Strict Thank You Block ---
#         clean_text = t_low.replace(".", "").replace("!", "").replace("?", "").strip()
#         if clean_text in ["thank you", "thanks", "thank you for watching", "thanks for watching", "bye"]:
#              print(f"[Filter] Removed hallucination: '{t}'")
#              return None

#         technical = ["subtitle", "copyright", "amara", "community"]
#         if any(x in t_low for x in technical): return None

#         valid_shorts = ["yes", "no", "yeah", "yep", "nope", "hi", "hey", "ok", "okay"]
#         if len(t) < 5 and clean_text not in valid_shorts: return None
#         if not any(ch.isalpha() for ch in t): return None
#         return t

#     def _record_and_predict_worker(self):
#         try:
#             sr = FEATURE_SETTINGS.get("sample_rate", 16000)
#             dur = float(self.calib.get("record_seconds", 3.0)) 
#             use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
#             print(f"[Audio] use_pepper_mic={use_pepper}")

#             if use_pepper:
#                 try:
#                     raw = self.pepper.record(seconds=int(max(1, round(dur))), mode=PEPPER.get("record_mode", "seconds"))
#                     y, sr_file = _bytes_to_audio(raw, sr_hint=48000)
#                     if sr_file != sr: y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
#                     print("[Audio] source: Pepper mic")
#                 except Exception as e:
#                     traceback.print_exc(); y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()
#             else:
#                 y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()

#             # 1. Gate Check
#             if not self._has_speech(y, sr):
#                 self._add_msg_safe("🎤 (no speech)", is_user=True)
#                 msg = "I couldn't hear any speech. Please try again a bit closer."
#                 self._add_msg_safe(msg, is_user=False)
#                 self.sig_update_emoji.emit("neutral") # Signal!
#                 self._say(msg); return

#             # 2. Amplitude Check
#             if float(np.max(np.abs(y))) < float(self.calib.get("min_amp", 0.10)):
#                 self._add_msg_safe("🎤 (quiet)", is_user=True)
#                 reply = RESPONSES.get("Uncertain", "I couldn't hear clearly.")
#                 self._add_msg_safe(reply, is_user=False)
#                 self.sig_update_emoji.emit("neutral") # Signal!
#                 self._say(reply); return

#             self._add_msg_safe("🎤 (audio captured)", is_user=True)
#             self.sig_update_emoji.emit("thinking") # Signal!

#             # 3. SER (Emotion First)
#             feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
#             x = feats[np.newaxis, :, :].astype("float32")
#             logits = self.session.run(None, {self.input_name: x})[0][0]
#             T = max(1e-6, float(self.temperature)); probs = softmax(logits / T)
#             label = self._decode(probs)

#             # DEBUG: see what the model is actually doing
#             print(f"[SER] label={label}, probs={probs}, classes={self.classes}")


#             # YOUR SAFETY SHIELD (Preserved)
#             g = self.last_gate_stats or {}
#             if label == "sad" and (g.get("p95", 1.0) < 0.035 or g.get("rms", 1.0) < 0.020):
#                 print("[Safety] overriding sad -> Uncertain due to low-energy audio")
#                 label = "Uncertain"

#             self.sig_update_emoji.emit(label) # Update face now

#             if self._should_decay_lock() and label in ("happy", "sad"): self._lock_emotion(label)

#             # 4. ASR (Text)
#             transcript = None
#             try:
#                 raw_txt = self.asr.transcribe(y, sr)
#                 print(f"[ASR raw] {raw_txt!r}")          # DEBUG 1: raw Whisper text

#                 transcript = self._clean_transcript(raw_txt)
#                 print(f"[ASR clean] {transcript!r}")     # DEBUG 2: after your filters

#             except Exception: transcript = None
#             finally: self._add_msg_safe(transcript if transcript else "🎤 (no transcript)", is_user=True)

#             # 5. Presence Mode (If text missing) -> Logic Fixed here!
#             # If text is missing, we RESET to Neutral and say "Couldn't hear".
#             if not transcript:
#                 reply = RESPONSES.get(
#                     "Uncertain",
#                     "I couldn't quite catch any words. Let's try again."
#                 )
#                 self._add_msg_safe(reply, is_user=False)
#                 self.sig_update_emoji.emit("neutral") # Reset to neutral
#                 self._say(reply)
#                 return

#             # 6. Chat Logic
#             self._maybe_override_from_text(transcript)
#             emotion_for_llm = self.emotion_locked or (label if label in ("happy", "sad") else "unknown")

#             intent_direct = False
#             t_low = transcript.lower()
#             for c in ["how can i", "how do i", "what should i", "tips", "suggest"]:
#                 if c in t_low: intent_direct = True; break
            
#             if self.dialog_phase == "opener":
#                 reply = RESPONSES.get(emotion_for_llm, RESPONSES.get("Uncertain", "I am not sure how you are feeling."))
#                 self.dialog_phase = "chat"
#             elif intent_direct:
#                 reply = self.chat.reply(emotion_for_llm, transcript)
#                 self.dialog_phase = "chat"
#             else:
#                 reply = self.chat.reply(emotion_for_llm, transcript)

#             self.turns_since_lock += 1
#             self._add_msg_safe(reply, is_user=False)
#             self._say(reply)
#             self.on_prediction(label)

#             # Save last audio chunk for offline debugging
#             try:
#                 sf.write("debug_pepper.wav", y, sr, subtype="PCM_16")
#                 print(f"[DEBUG] wrote debug_pepper.wav, shape={y.shape}, sr={sr}")
#             except Exception as e:
#                 print("[DEBUG] failed to write debug_pepper.wav:", e)


#         except Exception:
#             traceback.print_exc(); self._add_msg_safe("Something went wrong.", is_user=False); self._finish()

#     def reset_session(self):
#         self.emotion_locked = None; self.emotion_locked_at = None; self.turns_since_lock = 0
#         self.chat.history = []; self.dialog_phase = "opener"
#         self._add_msg_safe("🆕 New session started.", is_user=False)

# # ---- Run ----
# if __name__ == "__main__":
#     print("[BOOT] OLLAMA_HOST =", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
#     app = QApplication(sys.argv); w = EmotionApp(); w.show(); sys.exit(app.exec_())



# final1

# # gui_live_predict.py
# import sys, os, json, threading, traceback, tempfile, time
# import numpy as np
# import librosa, sounddevice as sd, pyttsx3
# import soundfile as sf
# import requests
# import onnxruntime as ort
# from io import BytesIO
# from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
# from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QPushButton, QLabel,
#     QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
# )
# from extract_features import extract_mfcc

# # --- Project modules ---
# from config import FEATURE_SETTINGS, RESPONSES, PEPPER
# try:
#     from config import ASSISTANT_STYLE
# except Exception:
#     ASSISTANT_STYLE = (
#         "You are a friendly, everyday wellbeing companion for mild support. "
#         "Keep replies warm, natural, and brief (1–2 short sentences). No diagnosis or clinical terms."
#     )

# try:
#     from pepper_client import PepperClient
# except Exception:
#     PepperClient = None

# # ==== Runtime tweaks (Windows OpenMP noise) ====
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# # ---------------- Files & loading ----------------
# MODELS_DIR = "models"
# TRACKS = [
#     # 1. SSL Model (Expects 768 dims)
#     {
#         "name": "SSL v1",  
#         "dir": os.path.join(MODELS_DIR, "ssl_v1"),  
#         "onnx": ["model_ssl.onnx"],                           
#         "type": "ssl"   # <--- CORRECT: Uses SSL frontend (768 dims)
#     },
#     # 2. MFCC Model (Expects 45 dims)
#     {
#         "name": "MFCC v1", 
#         "dir": os.path.join(MODELS_DIR, "mfcc_v1"), 
#         "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], 
#         "type": "mfcc"  # <--- CORRECT: Uses MFCC frontend (45 dims)
#     },
# ]

# # ---------------- Utils ----------------
# def softmax(x):
#     x = x - np.max(x); e = np.exp(x); return e / e.sum()

# def _speak_async(text, on_done):
#     def run():
#         try:
#             eng = pyttsx3.init()
#             eng.say(text); eng.runAndWait()
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

# # --- LOCAL ASR ---
# class LocalASR:
#     def __init__(self, model_size="small.en"): 
#         from faster_whisper import WhisperModel
#         self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

#     def transcribe(self, y, sr):
#         y = np.asarray(y, dtype=np.float32).reshape(-1)
#         if y.size == 0: return None

#         peak = float(np.max(np.abs(y)))
#         if peak > 0: y = (y / peak) * 0.9
#         pad = np.zeros(int(0.25 * sr), dtype=np.float32)
#         y = np.concatenate([pad, y, pad], axis=0)

#         fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
#         try:
#             sf.write(path, y, sr, subtype="PCM_16")
#             segments, _ = self.model.transcribe(
#                 path, language="en", vad_filter=False, beam_size=1
#             )
#             text = " ".join(s.text for s in segments).strip()
#             return text or None
#         finally:
#             try: os.remove(path)
#             except Exception: pass

# class ChatEngine:
#     def __init__(self, model=None, host=None, debug=True):
#         self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#         self.host  = (host  or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
#         self.url_chat = f"{self.host}/api/chat"
#         self.url_gen  = f"{self.host}/api/generate"
#         self.system = ASSISTANT_STYLE
#         self.history = []
#         self.debug = debug
#         try:
#             requests.get(f"{self.host}/api/tags", timeout=5)
#         except Exception as e:
#             print(f"[Ollama] ping failed: {e}")

#     def _make_prompt(self, emotion: str, transcript: str | None) -> str:
#         t = (transcript or "").strip()
#         return (
#             f"{self.system}\n"
#             f"UserEmotion: {emotion or 'unknown'}\n"
#             f"UserSaid: {t if t else '(empty)'}\n"
#             "If UserSaid looks like a question, answer it directly with 1–2 concrete suggestions. "
#             "Otherwise reply in 1–2 short, natural sentences."
#         )

#     def _compose_text_prompt(self, prompt: str) -> str:
#         parts = [f"System: {self.system}"]
#         for m in self.history[-8:]:
#             parts.append(f"{m['role'].capitalize()}: {m['content']}")
#         parts.append(f"User: {prompt}")
#         parts.append("Assistant:")
#         return "\n".join(parts)

#     def reply(self, emotion: str, transcript: str | None) -> str:
#         prompt = self._make_prompt(emotion, transcript)
#         try:
#             text_prompt = self._compose_text_prompt(prompt)
#             print(f"[Ollama] POST {self.url_gen} model={self.model}")
#             r = requests.post(
#                 self.url_gen,
#                 json={
#                     "model": self.model, "prompt": text_prompt, "stream": False,
#                     "options": {"temperature": 0.6, "repeat_penalty": 1.15, "num_predict": 90},
#                 },
#                 timeout=60, 
#             )
#             r.raise_for_status()
#             text = (r.json().get("response") or "").strip()
#             if not text: raise RuntimeError("empty generate response")
#             self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
#             self.history = self.history[-12:]
#             return text
#         except Exception as e:
#             if self.debug: print("[Ollama error]", repr(e))
#             return "I’m having a hiccup reaching my language model. We can keep talking, or try again shortly."

# # ---------------- Chat UI ----------------
# class ChatBubble(QLabel):
#     def __init__(self, text, is_user=False):
#         super().__init__(text)
#         self.setWordWrap(True)
#         self.setMaximumWidth(360)
#         color = "#e0e0e0" if is_user else "#95abbe"
#         self.setStyleSheet(f"background:{color}; border-radius:10px; padding:8px;")

# # ---------------- Main App ----------------
# class EmotionApp(QWidget):
#     sig_add_msg = pyqtSignal(str, bool)
#     sig_finish  = pyqtSignal()
#     sig_update_emoji = pyqtSignal(str)

#     def __init__(self):
#         super().__init__()
#         self._last_click = 0.0
#         self.setWindowTitle("Speech Emotion Detection Application")
#         self.setGeometry(100, 100, 400, 500)
#         self.emotion_locked = None
#         self.emotion_locked_at = None
#         self.turns_since_lock = 0
#         self.last_gate_stats = None
        
#         self.sig_update_emoji.connect(self.update_emoji)

#         # top bar
#         self.mic_icon = QLabel()
#         self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.mic_icon.setPixmap(self.mic_off)

#         # --- EMOJI UI ---
#         self.emoji_label = QLabel("😐")
#         self.emoji_label.setAlignment(Qt.AlignCenter)
#         self.emoji_label.setStyleSheet("font-size: 72px; color: #888888;")

#         # Chat area
#         self.chat_layout = QVBoxLayout()
#         self.chat_layout.setAlignment(Qt.AlignTop)
#         container = QWidget(); container.setLayout(self.chat_layout)
#         scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container)
#         self.scroll = scroll

#         # Record button
#         self.button = QPushButton("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setIconSize(QSize(20, 20))
#         self.button.clicked.connect(self.record_and_predict)

#         # Layout
#         main = QVBoxLayout(self)
#         main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
#         main.addWidget(self.emoji_label, alignment=Qt.AlignCenter)
#         main.addWidget(scroll)
#         main.addWidget(self.button)

#         # Runtime state
#         self.is_processing = False
#         self.session = None
#         self.input_name = None
#         self.model_type = None
#         self.classes = ["happy", "sad"]
#         self.calib = {"mode":"threshold","sad_threshold":0.57,"min_confidence":0.50,"min_amp":0.1,"record_seconds":3.0}
#         self.temperature = 1.0
#         self.ssl = None

#         # Engines
#         self.asr  = LocalASR()
#         self.chat = ChatEngine(debug=True)
#         self.dialog_phase = "opener"
#         self.sig_add_msg.connect(self._add_msg)
#         self.sig_finish.connect(self._finish_ui)

#         self._auto_load_model()
#         self.pepper = None
#         if PEPPER.get("enabled") and PepperClient:
#             try:
#                 self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
#                 self.pepper.connect()
#                 print("[Pepper] connected. Flushing buffer...")
#                 try:
#                     self.pepper.record(seconds=2, mode="seconds")
#                     print("[Pepper] Buffer flushed.")
#                 except Exception: pass
#             except Exception as e:
#                 print("[Pepper] connect failed:", e)
#                 self.pepper = None

#     def update_emoji(self, emotion):
#         if emotion == "happy":
#             self.emoji_label.setText("😃"); self.emoji_label.setStyleSheet("font-size: 72px; color: #2ecc71;")
#         elif emotion == "sad":
#             self.emoji_label.setText("😔"); self.emoji_label.setStyleSheet("font-size: 72px; color: #e74c3c;")
#         elif emotion == "listening":
#             self.emoji_label.setText("🙂"); self.emoji_label.setStyleSheet("font-size: 72px; color: #3498db;")
#         elif emotion == "thinking":
#             self.emoji_label.setText("🤔"); self.emoji_label.setStyleSheet("font-size: 72px; color: #f1c40f;")
#         else:
#             self.emoji_label.setText("😐"); self.emoji_label.setStyleSheet("font-size: 72px; color: #888888;")

#     def _has_speech(self, y, sr):
#         # Tighter Gate Logic (Blocks Fan Noise)
#         frame_len = int(sr * 0.02); hop_len = int(sr * 0.01)
#         rmse = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        
#         rms_val = float(np.max(rmse)) if len(rmse) > 0 else 0.0
#         peak_val = float(np.max(np.abs(y)))
        
#         # Threshold: 0.035 blocks fan noise, allows quiet speech
#         if rms_val < 0.035:
#             print(f"[Gate] rms={rms_val:.4f} peak={peak_val:.4f} -> BLOCK (Too Quiet)")
#             return False
            
#         print(f"[Gate] rms={rms_val:.4f} peak={peak_val:.4f} -> PASS")
#         self.last_gate_stats = {"rms": rms_val, "peak": peak_val}
#         return True

#     def _should_decay_lock(self, minutes=7, max_turns=4):
#         if self.emotion_locked is None: return True
#         if self.emotion_locked_at and (time.time() - self.emotion_locked_at) > minutes * 60: return True
#         if self.turns_since_lock >= max_turns: return True
#         return False

#     def _lock_emotion(self, label):
#         self.emotion_locked = label
#         self.emotion_locked_at = time.time()
#         self.turns_since_lock = 0

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

#     def _add_msg(self, text, is_user=False):
#         bubble = ChatBubble(text, is_user)
#         row = QHBoxLayout()
#         if is_user: row.addStretch(); row.addWidget(bubble)
#         else: row.addWidget(bubble); row.addStretch()
#         f = QFrame(); f.setLayout(row); self.chat_layout.addWidget(f)
#         self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

#     def _add_msg_safe(self, text, is_user=False): self.sig_add_msg.emit(text, is_user)

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
#             temp = _read_json(os.path.join(track["dir"], "temperature.json"), {"temperature":1.0})
#             self.temperature = float(temp.get("temperature", 1.0))
#             self.classes = _read_json(calib_path, {}).get("classes", self.classes)

#             so = ort.SessionOptions()
#             so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.log_severity_level = 3
#             self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
#             self.input_name = self.session.get_inputs()[0].name
            
#             # --- INPUT DIMENSION SAFETY CHECK ---
#             inp_shape = self.session.get_inputs()[0].shape
#             print(f"[BOOT] Model Input Shape: {inp_shape}")
            
#             # Check for Mismatch: Expected 45 but got SSL (768)
#             if self.model_type == "ssl" and inp_shape[-1] == 45:
#                 print("!! WARNING: You selected SSL, but model expects 45 dims. Switching to MFCC features !!")
#                 self.model_type = "mfcc"
            
#             # Check for Mismatch: Expected 768 but got MFCC (45)
#             if self.model_type == "mfcc" and inp_shape[-1] == 768:
#                 print("!! WARNING: You selected MFCC, but model expects 768 dims. Switching to SSL features !!")
#                 self.model_type = "ssl"

#             if self.model_type == "ssl" and self.ssl is None:
#                 import warnings; warnings.simplefilter("ignore")
#                 from ssl_frontend import SSLFrontend; self.ssl = SSLFrontend()
#         except Exception:
#             traceback.print_exc()
#             self._add_msg_safe("Setup problem.", is_user=False)

#     def _feat_mfcc(self, y, sr):
#         if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr): y = yt
#         except Exception: pass
#         return extract_mfcc(array=y, sr=sr)

#     def _feat_ssl(self, y, sr):
#         if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
#         try:
#             yt, _ = librosa.effects.trim(y, top_db=30)
#             if len(yt) > int(0.25 * sr): y = yt
#         except Exception: pass
#         return self.ssl(y)

#     def _decode(self, probs):
#         try: idx_h = self.classes.index("happy"); idx_s = self.classes.index("sad")
#         except ValueError: idx_h, idx_s = 0, 1
#         p_h, p_s = float(probs[idx_h]), float(probs[idx_s])
#         if max(p_h, p_s) < float(self.calib.get("min_confidence", 0.50)): return "Uncertain"
#         if self.calib.get("mode", "threshold") == "threshold":
#             return "sad" if p_s >= float(self.calib.get("sad_threshold", 0.57)) else "happy"
#         return "happy" if p_h >= p_s else "sad"

#     def _finish_ui(self):
#         self.mic_icon.setPixmap(self.mic_off)
#         self.button.setText("🎤  Record & Detect")
#         self.button.setIcon(QIcon("mic-off.png"))
#         self.button.setEnabled(True)
#         self.is_processing = False

#     def _finish(self): self.sig_finish.emit()

#     def record_and_predict(self):
#         now = time.monotonic()
#         if now - self._last_click < 0.8: return
#         self._last_click = now
#         if self.is_processing or self.session is None: return
#         self.is_processing = True
#         self.button.setEnabled(False)
#         self.button.setText("🔴  Listening…")
#         self.button.setIcon(QIcon("mic-on.png"))
#         self.mic_icon.setPixmap(self.mic_on)
        
#         self.sig_update_emoji.emit("listening") 
#         QApplication.processEvents()
#         threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

#     def _clean_transcript(self, s: str | None) -> str | None:
#         if not s: return None
#         t = s.strip(); t_low = t.lower()
#         if not t: return None

#         clean_text = t_low.replace(".", "").replace("!", "").replace("?", "").strip()
#         if clean_text in ["thank you", "thanks", "thank you for watching", "bye"]:
#              print(f"[Filter] Removed hallucination: '{t}'")
#              return None

#         technical = ["subtitle", "copyright", "amara", "community"]
#         if any(x in t_low for x in technical): return None

#         if len(t) < 5 and clean_text not in ["yes", "no", "hi", "hey", "ok"]: return None
#         if not any(ch.isalpha() for ch in t): return None
#         return t

#     def _record_and_predict_worker(self):
#         try:
#             # 0. Record
#             sr = FEATURE_SETTINGS.get("sample_rate", 16000)
#             dur = float(self.calib.get("record_seconds", 3.0)) 
#             use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
#             print(f"[Audio] use_pepper_mic={use_pepper}")

#             if use_pepper:
#                 try:
#                     raw = self.pepper.record(seconds=int(max(1, round(dur))), mode=PEPPER.get("record_mode", "seconds"))
#                     y, sr_file = _bytes_to_audio(raw, sr_hint=48000)
#                     if sr_file != sr: y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
#                     print("[Audio] source: Pepper mic")
#                 except Exception as e:
#                     traceback.print_exc(); y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()
#             else:
#                 y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()

#             # 1. Gate Check
#             if not self._has_speech(y, sr):
#                 self._add_msg_safe("🎤 (too quiet)", is_user=True)
#                 msg = "I couldn't hear you. Please speak a little louder."
#                 self._add_msg_safe(msg, is_user=False)
#                 self.sig_update_emoji.emit("neutral") 
#                 self._say(msg); return

#             self._add_msg_safe("🎤 (processing...)", is_user=True)
#             self.sig_update_emoji.emit("thinking")

#             # 2. Transcribe
#             transcript = None
#             try:
#                 raw_txt = self.asr.transcribe(y, sr)
#                 print(f"[ASR raw] {raw_txt!r}")

#                 transcript = self._clean_transcript(raw_txt)
#                 print(f"[ASR clean] {transcript!r}")
#             except Exception: transcript = None

#             # 3. No Text = No Emotion (Safety)
#             if not transcript or len(transcript.strip()) < 2:
#                 msg = "I heard a noise, but I couldn't understand the words."
#                 self._add_msg_safe(msg, is_user=False)
#                 self.sig_update_emoji.emit("neutral") 
#                 self._say(msg); return

#             self._add_msg_safe(f"You: {transcript}", is_user=True)

#             # 4. Audio Emotion
#             feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
#             x = feats[np.newaxis, :, :].astype("float32")
#             logits = self.session.run(None, {self.input_name: x})[0][0]
#             T = max(1e-6, float(self.temperature)); probs = softmax(logits / T)
#             audio_label = self._decode(probs)
#             print(f"[AI] Audio thinks: {audio_label}")

#             # 5. TEXT OVERRIDE (Brain over Ear)
#             final_label = audio_label
#             text_lower = transcript.lower()
#             if any(w in text_lower for w in ["happy", "good", "great", "joy", "excited", "love"]):
#                 final_label = "happy"
#                 print(f"[Override] Text found happy words -> Force HAPPY")
#             elif any(w in text_lower for w in ["sad", "down", "depressed", "bad", "unhappy", "cry"]):
#                 final_label = "sad"
#                 print(f"[Override] Text found sad words -> Force SAD")
            
#             print(f"[Final] Emotion used: {final_label}")   

#             # 6. Response
#             self.sig_update_emoji.emit(final_label)

#             if self.dialog_phase == "opener":
#                 reply = RESPONSES.get(final_label, "I am listening.")
#                 self.dialog_phase = "chat"
#             else:
#                 reply = self.chat.reply(final_label, transcript)

#             self._add_msg_safe(reply, is_user=False)
#             self._say(reply)
#             self.on_prediction(final_label)

#             try:
#                 sf.write("debug_pepper.wav", y, sr, subtype="PCM_16")
#                 print(f"[DEBUG] wrote debug_pepper.wav, shape={y.shape}, sr={sr}")
#             except Exception as e:
#                 print("[DEBUG] failed to write debug_pepper.wav:", e)


#         except Exception:
#             traceback.print_exc(); self._add_msg_safe("Something went wrong.", is_user=False); self._finish()

#     def reset_session(self):
#         self.emotion_locked = None; self.emotion_locked_at = None; self.turns_since_lock = 0
#         self.chat.history = []; self.dialog_phase = "opener"
#         self._add_msg_safe("🆕 New session started.", is_user=False)

# # ---- Run ----
# if __name__ == "__main__":
#     print("[BOOT] OLLAMA_HOST =", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
#     app = QApplication(sys.argv); w = EmotionApp(); w.show(); sys.exit(app.exec_())




#final 2.1
# gui_live_predict.py
import sys, os, json, threading, traceback, tempfile, time
import numpy as np
import librosa, sounddevice as sd, pyttsx3
import soundfile as sf
import requests
import onnxruntime as ort
from io import BytesIO
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QFont,QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, QSizePolicy
)
from extract_features import extract_mfcc

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES, PEPPER
try:
    from config import ASSISTANT_STYLE
except Exception:
    ASSISTANT_STYLE = (
        "You are a friendly, everyday wellbeing companion for mild support. "
        "Keep replies warm, natural, and brief (1–2 short sentences). No diagnosis or clinical terms."
    )

try:
    from pepper_client import PepperClient
except Exception:
    PepperClient = None

try:
    # Initialize ONCE globally, not inside the function
    TTS_ENGINE = pyttsx3.init()
except Exception:
    TTS_ENGINE = None

# ==== Runtime tweaks (Windows OpenMP noise) ====
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ---------------- Files & loading ----------------
MODELS_DIR = "models"
TRACKS = [
    # 1. SSL Model (Expects 768 dims)
    {
        "name": "SSL v1",  
        "dir": os.path.join(MODELS_DIR, "ssl_v1"),  
        "onnx": ["model_ssl.onnx"],                           
        "type": "ssl"
    },
    # 2. MFCC Model (Expects 45 dims)
    {
        "name": "MFCC v1", 
        "dir": os.path.join(MODELS_DIR, "mfcc_v1"), 
        "onnx": ["model_mfcc.onnx", "model_mfcc_int8.onnx"], 
        "type": "mfcc"
    },
]

# ---------------- Utils ----------------
def softmax(x):
    x = x - np.max(x); e = np.exp(x); return e / e.sum()

# def _speak_async(text, on_done):
#     def run():
#         try:
#             eng = pyttsx3.init()
#             eng.say(text); eng.runAndWait()
#         except Exception:
#             traceback.print_exc()
#         finally:
#             QTimer.singleShot(0, on_done)
#     threading.Thread(target=run, daemon=True).start()

# --- SAFE TTS SETUP (Final 2.1) ---


def _speak_async(text, on_done):
    def run():
        try:
            if TTS_ENGINE:
                TTS_ENGINE.say(text)
                TTS_ENGINE.runAndWait()
        except Exception:
            traceback.print_exc()
        finally:
            # Always notify UI that speech is done, even if audio failed
            QTimer.singleShot(0, on_done)
    threading.Thread(target=run, daemon=True).start()

def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return default

def _find_existing(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

def _bytes_to_audio(raw: bytes, sr_hint: int = 16000):
    if len(raw) >= 12 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
        y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1: y = y[:, 0]
        return y, sr
    a = np.frombuffer(raw, dtype=np.int16)
    if a.size == 0: raise ValueError("Pepper returned empty audio payload")
    y = (a.astype(np.float32)) / 32768.0
    return y, sr_hint

# --- LOCAL ASR ---
class LocalASR:
    def __init__(self, model_size="small.en"): 
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, y, sr):
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0: return None

        peak = float(np.max(np.abs(y)))
        if peak > 0: y = (y / peak) * 0.9
        pad = np.zeros(int(0.25 * sr), dtype=np.float32)
        y = np.concatenate([pad, y, pad], axis=0)

        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        try:
            sf.write(path, y, sr, subtype="PCM_16")
            segments, _ = self.model.transcribe(
                path, language="en", vad_filter=False, beam_size=1
            )
            text = " ".join(s.text for s in segments).strip()
            return text or None
        finally:
            try: os.remove(path)
            except Exception: pass

class ChatEngine:
    def __init__(self, model=None, host=None, debug=True):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        self.host  = (host  or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.url_chat = f"{self.host}/api/chat"
        self.url_gen  = f"{self.host}/api/generate"
        self.system = ASSISTANT_STYLE
        self.history = []
        self.debug = debug
        try:
            requests.get(f"{self.host}/api/tags", timeout=5)
        except Exception as e:
            print(f"[Ollama] ping failed: {e}")

    def _make_prompt(self, emotion: str, transcript: str | None) -> str:
        t = (transcript or "").strip()
        return (
            f"{self.system}\n"
            f"UserEmotion: {emotion or 'unknown'}\n"
            f"UserSaid: {t if t else '(empty)'}\n"
            "If UserSaid looks like a question, answer it directly with 1–2 concrete suggestions. "
            "Otherwise reply in 1–2 short, natural sentences."
        )

    def _compose_text_prompt(self, prompt: str) -> str:
        parts = [f"System: {self.system}"]
        for m in self.history[-8:]:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def reply(self, emotion: str, transcript: str | None) -> str:
        prompt = self._make_prompt(emotion, transcript)
        try:
            text_prompt = self._compose_text_prompt(prompt)
            print(f"[Ollama] POST {self.url_gen} model={self.model}")
            r = requests.post(
                self.url_gen,
                json={
                    "model": self.model, "prompt": text_prompt, "stream": False,
                    "options": {"temperature": 0.6, "repeat_penalty": 1.15, "num_predict": 90},
                },
                timeout=60, 
            )
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            if not text: raise RuntimeError("empty generate response")
            self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
            self.history = self.history[-12:]
            return text
        except Exception as e:
            if self.debug: print("[Ollama error]", repr(e))
            return "I’m having a hiccup reaching my language model. We can keep talking, or try again shortly."

# ---------------- Chat UI Elements ----------------
class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        self.setMargin(10)
        font = QFont("Segoe UI", 10)
        self.setFont(font)
        
        if is_user:
            # User: Greenish/Gray bubble, right aligned
            self.setStyleSheet("background-color: #DCF8C6; color: black; border-radius: 12px; padding: 6px;")
        else:
            # Bot: White/Gray bubble, left aligned
            self.setStyleSheet("background-color: #EAEAEA; color: black; border-radius: 12px; padding: 6px;")

class EmojiResult(QLabel):
    """A special centered widget for the emotion result inside the chat."""
    def __init__(self, emotion):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: transparent;")
        
        # Select Emoji
        if emotion == "happy":
            self.setText("😃")
            self.setToolTip("Happy Detected")
        elif emotion == "sad":
            self.setText("😔")
            self.setToolTip("Sad Detected")
        else:
            self.setText("😐")
            
        # Large Font for the emoji
        font = QFont("Segoe UI Emoji", 32) 
        self.setFont(font)

# ---------------- Main App ----------------
class EmotionApp(QWidget):
    sig_add_msg = pyqtSignal(str, bool)     # Add text message
    sig_finish  = pyqtSignal()
    sig_add_emoji = pyqtSignal(str)         # Add emoji bubble to chat
    sig_update_status = pyqtSignal(str)     # Update top status icon

    def __init__(self):
        super().__init__()
        self._last_click = 0.0
        self.setWindowTitle("Speech Emotion Detection Application")
        self.setGeometry(100, 100, 420, 600)
        self.emotion_locked = None
        self.emotion_locked_at = None
        self.turns_since_lock = 0
        self.last_gate_stats = None
        
        # Connect Signals
        self.sig_add_msg.connect(self._add_msg)
        self.sig_add_emoji.connect(self._add_emoji_bubble)
        self.sig_update_status.connect(self.update_status_icon)
        self.sig_finish.connect(self._finish_ui)

        # --- Top Bar (Status) ---
        top_bar = QHBoxLayout()
        # Mic Icon
        self.mic_icon = QLabel()
        # --- FINAL 2.1: SAFE ICON LOADING ---
        def load_icon(path, color):
            if os.path.exists(path):
                return QPixmap(path).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                # Create a simple colored square if image is missing
                pix = QPixmap(24, 24)
                pix.fill(QColor(color)) 
                return pix
            
        # Ensure you import QColor at the top: from PyQt5.QtGui import QColor
        self.mic_off = load_icon("mic-off.png", "#808080") # Grey square fallback
        self.mic_on  = load_icon("mic-on.png",  "#FF0000") # Red square fallback
        self.mic_icon.setPixmap(self.mic_off)

        # Status Emoji (The "Agent State")
        self.status_label = QLabel("😐")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 40px; color: #555555;")
        self.status_label.setToolTip("Status: Idle")

        top_bar.addStretch()
        top_bar.addWidget(self.mic_icon)
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()

        # Chat area
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10) # Space between bubbles

        container = QWidget()
        container.setLayout(self.chat_layout)
        container.setStyleSheet("background-color: #FFFFFF;")
        
        scroll = QScrollArea() 
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet("border: none;")
        self.scroll = scroll

        # Record button
        self.button = QPushButton("🎤  Record & Detect")
        self.button.setIcon(QIcon("mic-off.png"))
        self.button.setIconSize(QSize(20, 20))
        self.button.setMinimumHeight(50)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7; 
                color: white; 
                font-size: 16px; 
                border-radius: 8px;
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
            }
        """)
        self.button.clicked.connect(self.record_and_predict)

        # Layout
        main = QVBoxLayout(self)
        main.addLayout(top_bar)
        main.addWidget(scroll)
        main.addWidget(self.button)

        # Runtime state
        self.is_processing = False
        self.session = None
        self.input_name = None
        self.model_type = None
        self.classes = ["happy", "sad"]
        self.calib = {"mode":"threshold","sad_threshold":0.57,"min_confidence":0.50,"min_amp":0.1,"record_seconds":3.0}
        self.temperature = 1.0
        self.ssl = None

        # Engines
        self.asr  = LocalASR()
        self.chat = ChatEngine(debug=True)
        self.dialog_phase = "opener"

        self._auto_load_model()
        self.pepper = None
        if PEPPER.get("enabled") and PepperClient:
            try:
                self.pepper = PepperClient(PEPPER["ip"], PEPPER["port"])
                self.pepper.connect()
                print("[Pepper] connected. Flushing buffer...")
                try:
                    self.pepper.record(seconds=2, mode="seconds")
                    print("[Pepper] Buffer flushed.")
                except Exception: pass
            except Exception as e:
                print("[Pepper] connect failed:", e)
                self.pepper = None
    
    # --- UI Updaters ---
    def update_status_icon(self, state):
        """Updates the top status icon based on agent state."""
        if state == "listening":
            self.status_label.setText("🙂") # Listening face
            self.status_label.setStyleSheet("font-size: 40px; color: #3498db;") # Blue
        elif state == "thinking":
            self.status_label.setText("🤔") # Thinking face
            self.status_label.setStyleSheet("font-size: 40px; color: #f1c40f;") # Yellow
        else: # idle
            self.status_label.setText("😐") # Neutral/Idle
            self.status_label.setStyleSheet("font-size: 40px; color: #888888;") # Grey

    def _add_emoji_bubble(self, emotion):
        """Inserts a large centered emoji into the chat stream."""
        emoji_widget = EmojiResult(emotion)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(emoji_widget)
        row.addStretch()
        self.chat_layout.addLayout(row)
        self._scroll_to_bottom()
    
    def _scroll_to_bottom(self):
        # Allow layout to calculate new size before scrolling
        QApplication.processEvents()
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def _has_speech(self, y, sr):
        # Tighter Gate Logic (Blocks Fan Noise)
        frame_len = int(sr * 0.02); hop_len = int(sr * 0.01)
        rmse = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        
        rms_val = float(np.max(rmse)) if len(rmse) > 0 else 0.0
        peak_val = float(np.max(np.abs(y)))
        
        # Threshold: 0.035 blocks fan noise, allows quiet speech
        if rms_val < 0.035:
            print(f"[Gate] rms={rms_val:.4f} peak={peak_val:.4f} -> BLOCK (Too Quiet)")
            return False
            
        print(f"[Gate] rms={rms_val:.4f} peak={peak_val:.4f} -> PASS")
        self.last_gate_stats = {"rms": rms_val, "peak": peak_val}
        return True

    def _should_decay_lock(self, minutes=7, max_turns=4):
        if self.emotion_locked is None: return True
        if self.emotion_locked_at and (time.time() - self.emotion_locked_at) > minutes * 60: return True
        if self.turns_since_lock >= max_turns: return True
        return False

    def _lock_emotion(self, label):
        self.emotion_locked = label
        self.emotion_locked_at = time.time()
        self.turns_since_lock = 0

    def _say(self, text):
        def finish_on_ui(): QTimer.singleShot(0, self._finish)
        if self.pepper:
            def run():
                try: self.pepper.tts(text)
                except Exception: traceback.print_exc()
                finally: finish_on_ui()
            threading.Thread(target=run, daemon=True).start()
        else:
            _speak_async(text, finish_on_ui)

    def on_prediction(self, label: str): pass

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

    def _add_msg_safe(self, text, is_user=False): self.sig_add_msg.emit(text, is_user)

    def _auto_load_model(self):
        try:
            chosen = None
            for t in TRACKS:
                if onnx_path := _find_existing([os.path.join(t["dir"], fn) for fn in t["onnx"]]):
                    chosen = (t, onnx_path); break
            if not chosen: raise FileNotFoundError("No ONNX model found.")
            track, onnx_path = chosen
            self.model_type = track["type"]
            print(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")

            calib_path = os.path.join(track["dir"], "calibration.json")
            self.calib.update(_read_json(calib_path, {}))
            temp = _read_json(os.path.join(track["dir"], "temperature.json"), {"temperature":1.0})
            self.temperature = float(temp.get("temperature", 1.0))
            self.classes = _read_json(calib_path, {}).get("classes", self.classes)

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.log_severity_level = 3
            self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            
            inp_shape = self.session.get_inputs()[0].shape
            if self.model_type == "ssl" and inp_shape[-1] == 45:
                print("!! WARNING: Switching SSL -> MFCC features due to dimensions.")
                self.model_type = "mfcc"
            if self.model_type == "mfcc" and inp_shape[-1] == 768:
                print("!! WARNING: Switching MFCC -> SSL features due to dimensions.")
                self.model_type = "ssl"

            if self.model_type == "ssl" and self.ssl is None:
                import warnings; warnings.simplefilter("ignore")
                from ssl_frontend import SSLFrontend; self.ssl = SSLFrontend()
        except Exception:
            traceback.print_exc()
            self._add_msg_safe("Setup problem.", is_user=False)

    def _feat_mfcc(self, y, sr):
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
        try:
            yt, _ = librosa.effects.trim(y, top_db=30)
            if len(yt) > int(0.25 * sr): y = yt
        except Exception: pass
        return extract_mfcc(array=y, sr=sr)

    def _feat_ssl(self, y, sr):
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
        try:
            yt, _ = librosa.effects.trim(y, top_db=30)
            if len(yt) > int(0.25 * sr): y = yt
        except Exception: pass
        return self.ssl(y)

    def _decode(self, probs):
        try: idx_h = self.classes.index("happy"); idx_s = self.classes.index("sad")
        except ValueError: idx_h, idx_s = 0, 1
        p_h, p_s = float(probs[idx_h]), float(probs[idx_s])
        if max(p_h, p_s) < float(self.calib.get("min_confidence", 0.50)): return "Uncertain"
        if self.calib.get("mode", "threshold") == "threshold":
            return "sad" if p_s >= float(self.calib.get("sad_threshold", 0.57)) else "happy"
        return "happy" if p_h >= p_s else "sad"

    def _finish_ui(self):
        self.mic_icon.setPixmap(self.mic_off)
        self.button.setText("🎤  Record & Detect")
        self.button.setIcon(QIcon("mic-off.png"))
        self.button.setEnabled(True)
        self.is_processing = False
        # Reset Status to Idle
        self.sig_update_status.emit("idle")

    def _finish(self): self.sig_finish.emit()

    def record_and_predict(self):
        now = time.monotonic()
        if now - self._last_click < 0.8: return
        self._last_click = now
        if self.is_processing or self.session is None: return
        self.is_processing = True
        self.button.setEnabled(False)
        self.button.setText("🔴  Listening…")
        self.button.setIcon(QIcon("mic-on.png"))
        self.mic_icon.setPixmap(self.mic_on)
        
        self.sig_update_status.emit("listening")
        QApplication.processEvents()
        threading.Thread(target=self._record_and_predict_worker, daemon=True).start()

    def _clean_transcript(self, s: str | None) -> str | None:
        if not s: return None
        t = s.strip(); t_low = t.lower()
        if not t: return None

        clean_text = t_low.replace(".", "").replace("!", "").replace("?", "").strip()
        if clean_text in ["thank you", "thanks", "thank you for watching", "bye"]:
             print(f"[Filter] Removed hallucination: '{t}'")
             return None

        technical = ["subtitle", "copyright", "amara", "community"]
        if any(x in t_low for x in technical): return None
        
        # 3. ALLOW SHORT EMOTION WORDS (Fixes "Sad", "Mad", "Joy")
        allowed_short = ["yes", "no", "hi", "hey", "ok", "sad", "bad", "mad", "joy", "cry", "wow", "fun"]
        if len(clean_text) < 2: return None
        if len(clean_text) < 5 and clean_text not in allowed_short: 
            return None # Still filter "the" but keep "sad"

        if not any(ch.isalpha() for ch in t): return None
        return t

    def _record_and_predict_worker(self):
        try:
            # 0. Record
            sr = FEATURE_SETTINGS.get("sample_rate", 16000)
            dur = float(self.calib.get("record_seconds", 3.0)) 
            use_pepper = bool(self.pepper) and bool(PEPPER.get("use_pepper_mic", False))
            print(f"[Audio] use_pepper_mic={use_pepper}")

            if use_pepper:
                try:
                    raw = self.pepper.record(seconds=int(max(1, round(dur))), mode=PEPPER.get("record_mode", "seconds"))
                    y, sr_file = _bytes_to_audio(raw, sr_hint=48000)
                    if sr_file != sr: y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
                    print("[Audio] source: Pepper mic")
                except Exception as e:
                    traceback.print_exc(); y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()
            else:
                y = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype="float32"); sd.wait(); y = y.flatten()

            # 1. Gate Check
            if not self._has_speech(y, sr):
                self._add_msg_safe("🎤 (too quiet)", is_user=True)
                msg = "I couldn't hear you. Please speak a little louder."
                self.sig_update_status.emit("idle")
                self._say(msg); return

            self.sig_update_status.emit("thinking")

            # 2. Transcribe
            transcript = None
            try:
                raw_txt = self.asr.transcribe(y, sr)
                print(f"[ASR raw] {raw_txt!r}")
                transcript = self._clean_transcript(raw_txt)
                print(f"[ASR clean] {transcript!r}")
            except Exception: transcript = None

            # 3. No Text = No Emotion (Safety)
            if not transcript or len(transcript.strip()) < 2:
                msg = "I heard a noise, but I couldn't understand the words."
                self._add_msg_safe(msg, is_user=False)
                self.sig_update_status.emit("idle")
                self._say(msg)
                return

            self._add_msg_safe(f"You: {transcript}", is_user=True)

            # 4. Audio Emotion
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")
            logits = self.session.run(None, {self.input_name: x})[0][0]
            T = max(1e-6, float(self.temperature)); probs = softmax(logits / T)
            audio_label = self._decode(probs)
            print(f"[AI] Audio thinks: {audio_label}")

            # 5. TEXT OVERRIDE (Brain over Ear)
            final_label = audio_label
            text_lower = transcript.lower()
            if any(w in text_lower for w in ["happy", "good", "great", "joy", "excited", "love"]):
                final_label = "happy"
                print(f"[Override] Text found happy words -> Force HAPPY")
            elif any(w in text_lower for w in ["sad", "down", "depressed", "bad", "unhappy", "cry"]):
                final_label = "sad"
                print(f"[Override] Text found sad words -> Force SAD")
            
            print(f"[Final] Emotion used: {final_label}")   

            # 6. Response
            self.sig_add_emoji.emit(final_label)

            if self.dialog_phase == "opener":
                reply = RESPONSES.get(final_label, "I am listening.")
                self.dialog_phase = "chat"
            else:
                reply = self.chat.reply(final_label, transcript)

            self._add_msg_safe(reply, is_user=False)
            self._say(reply)
            self.on_prediction(final_label)

        except Exception:
            traceback.print_exc(); self._add_msg_safe("Something went wrong.", is_user=False); self._finish()

    def reset_session(self):
        self.emotion_locked = None; self.emotion_locked_at = None; self.turns_since_lock = 0
        self.chat.history = []; self.dialog_phase = "opener"
        self._add_msg_safe("🆕 New session started.", is_user=False)

# ---- Run ----
if __name__ == "__main__":
    print("[BOOT] OLLAMA_HOST =", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    app = QApplication(sys.argv); w = EmotionApp(); w.show(); sys.exit(app.exec_())