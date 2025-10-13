# gui_live_predict.py
import sys, os, json, threading, traceback
import numpy as np
import librosa, sounddevice as sd, pyttsx3

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame

# ---- TUNABLES (safer, more forgiving) ----
USE_ASR_GATE = True                 # set False to disable word-count gate
AMP_MIN = 0.08                      # min peak amplitude to consider "audible"
NON_SILENT_MIN_SEC = 0.9            # was 1.2 — easier to pass for short turns
SPLIT_TOP_DB = 30                   # was 28 — a bit less aggressive
ASR_MIN_WORDS = 2                   # was 3 — allow short phrases
sd.default.samplerate = 16000
sd.default.channels = 1

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES
from extract_features import extract_mfcc
# from ssl_frontend import SSLFrontend  # only if SSL model present

import onnxruntime as ort
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Dict
from scipy.io.wavfile import write as wavwrite

# ---------------- Files & loading ----------------
MODELS_DIR = "models"
TRACKS = [
    {"name": "SSL v1",  "dir": os.path.join(MODELS_DIR, "ssl_v1"),  "onnx": ["model_ssl.onnx"], "type": "ssl"},
    {"name": "MFCC v1", "dir": os.path.join(MODELS_DIR, "mfcc_v1"), "onnx": ["model_mfcc.onnx","model_mfcc_int8.onnx"], "type": "mfcc"},
]

def softmax(x):
    x = x - np.max(x); e = np.exp(x); return e / e.sum()

def _speak_async(text, on_done):
    def run():
        try:
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
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
        if os.path.exists(p): return p
    return None

def _load_secret_key(path="secrets.json"):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        full = os.path.join(base, path)
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f).get("GEMINI_API_KEY", "").strip()
    except Exception:
        return ""

def _supportive_fallback(user_text: str, emotion: str) -> str:
    s = (user_text or "").strip()
    if len(s) > 80: s = s[:80].rstrip() + "…"
    if emotion == "sad":   return f"I hear you about “{s}”. That sounds heavy. What part feels hardest right now?"
    if emotion == "happy": return f"That’s lovely to hear about “{s}”. What made it feel so good?"
    return "I want to make sure I understood. Could you say that again in a few words?"

# ---- Optional Vosk ASR gate ----
class LocalASR:
    def __init__(self, model_dir="models/vosk_en"):
        from vosk import Model
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Vosk model dir missing: {model_dir}")
        self.model = Model(model_dir)

    def transcribe(self, y: np.ndarray, sr: int) -> str:
        import numpy as np, librosa
        from vosk import KaldiRecognizer
        # 1) mono 16 kHz
        if y.ndim > 1:
            y = y[:,0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        # 2) pre-emphasis + trim + normalize + soft gate
        y = librosa.effects.preemphasis(y, zi=None, coef=0.97)
        yt, _ = librosa.effects.trim(y, top_db=25)
        if len(yt) > 0.25 * sr:
            y = yt
        y = librosa.util.normalize(y, axis=0)
        gate = 0.015
        y = np.where(np.abs(y) < gate, 0.0, y)
        # 3) Vosk
        rec = KaldiRecognizer(self.model, sr)
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
        step = 32000
        for i in range(0, len(pcm), step):
            rec.AcceptWaveform(pcm[i:i+step])
        import json as pyjson
        try:
            txt = pyjson.loads(rec.FinalResult()).get("text","").strip()
        except Exception:
            txt = ""
        return " ".join(txt.split())

# ---- Minimal chat memory ----
@dataclass
class ChatHistory:
    messages: List[Dict[str,str]] = field(default_factory=list)
    def add_user(self, t:str): self.messages.append({"role":"user","content":t})
    def add_assistant(self, t:str): self.messages.append({"role":"assistant","content":t})
    def as_prompt(self)->str:
        if not self.messages:
            return "Assistant:"
        return "\n".join(
            ("User: " + m["content"]) if m["role"]=="user" else ("Assistant: " + m["content"])
            for m in self.messages
        ) + "\nAssistant:"

# ---- Gemini client ----
class GeminiClient:
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise RuntimeError("Gemini API key missing. Add it to secrets.json.")
        genai.configure(api_key=api_key)
        self.model_name = model or "gemini-1.5-flash"

    def generate(self, system_prompt: str, emotion_hint: str, history: ChatHistory, max_tokens: int=220) -> str:
        # Construct model with system instruction
        gm = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=(
                system_prompt + "\n" + emotion_hint +
                "\nKeep replies supportive and concrete. Validate feelings. "
                "Ask at most one gentle follow-up. Avoid medical advice."
            )
        )
        # Relax safety so normal wellbeing talk isn't blocked silently
        safety = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_NONE"},
        ]
        try:
            resp = gm.generate_content(
                history.as_prompt(),
                generation_config={"max_output_tokens": max_tokens},
                safety_settings=safety
            )
            # Prefer resp.text, fallback to stitching parts
            text = getattr(resp, "text", "") or ""
            if not text and getattr(resp, "candidates", None):
                parts = getattr(resp.candidates[0].content, "parts", []) or []
                text = "".join(getattr(p, "text", "") for p in parts)
            return (text or "").strip()
        except Exception:
            traceback.print_exc()
            return ""

# ---------------- UI ----------------
class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        self.setStyleSheet(
            f"background:{'#e0e0e0' if is_user else '#95abbe'};"
            "border-radius:10px; padding:8px;"
        )

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Emotion Detection Application")
        self.setGeometry(100, 100, 400, 500)

        self.mic_icon = QLabel()
        self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mic_icon.setPixmap(self.mic_off)

        self.chat_layout = QVBoxLayout(); self.chat_layout.setAlignment(Qt.AlignTop)
        container = QWidget(); container.setLayout(self.chat_layout)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container); self.scroll = scroll

        self.button = QPushButton("🎤  Record & Detect")
        self.button.setIcon(QIcon("mic-off.png")); self.button.setIconSize(QSize(20, 20))
        self.button.clicked.connect(self.record_and_predict)

        main = QVBoxLayout(self)
        main.addWidget(self.mic_icon, alignment=Qt.AlignCenter)
        main.addWidget(scroll)
        main.addWidget(self.button)

        # runtime state
        self.is_processing = False
        self.session=None; self.input_name=None; self.model_type=None
        self.classes=["happy","sad"]
        self.calib={"mode":"threshold","sad_threshold":0.57,"min_confidence":0.50,"min_amp":0.02,"record_seconds":3.0}
        self.temperature=1.0; self.ssl=None

        self.has_started_chat=False; self.asr=None; self.llm=None; self.history=ChatHistory()
        self.stop_words={"stop","pause","that's all","thats all","no thanks","not now"}
        self.turn=0

        self._auto_load_model()

    def _add_msg(self, text, is_user=False):
        bubble = ChatBubble(text, is_user)
        row = QHBoxLayout()
        if is_user:
            row.addStretch(); row.addWidget(bubble)
        else:
            row.addWidget(bubble); row.addStretch()
        f = QFrame(); f.setLayout(row); self.chat_layout.addWidget(f)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def _auto_load_model(self):
        try:
            # Gemini
            try:
                api_key = _load_secret_key()
                if not api_key:
                    print("[WARN] Gemini API key missing (GEMINI_API_KEY not found in secrets.json)")
                self.llm = GeminiClient(model="gemini-1.5-flash", api_key=api_key)
                print("[BOOT] Gemini client ready")
            except Exception as e:
                print("[WARN] Gemini not ready:", e)
                self.llm = None

            # SER model
            chosen=None
            for t in TRACKS:
                onnx_path=_find_existing([os.path.join(t["dir"], fn) for fn in t["onnx"]])
                if onnx_path: chosen=(t,onnx_path); break
            if not chosen: raise FileNotFoundError("No ONNX model found in models/ssl_v1 or models/mfcc_v1.")
            track, onnx_path = chosen; self.model_type=track["type"]
            print(f"[BOOT] Emotion backend: {self.model_type.upper()} • {os.path.basename(onnx_path)}")

            calib_path=os.path.join(track["dir"], "calibration.json")
            self.calib.update(_read_json(calib_path, {}))
            temp=_read_json(os.path.join(track["dir"], "temperature.json"), {"temperature":1.0})
            self.temperature=float(temp.get("temperature",1.0))
            self.classes=_read_json(calib_path, {}).get("classes", self.classes)

            so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1; so.log_severity_level=3
            self.session=ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            self.input_name=self.session.get_inputs()[0].name

            # SSL frontend lazy-load if needed
            if self.model_type=="ssl" and self.ssl is None:
                import warnings
                try:
                    from transformers.utils import logging as hf_logging; hf_logging.set_verbosity_error()
                except Exception: pass
                warnings.filterwarnings("ignore", "Passing `gradient_checkpointing`.*")
                warnings.filterwarnings("ignore", "`clean_up_tokenization_spaces`.*")
                from ssl_frontend import SSLFrontend; self.ssl=SSLFrontend()

            # ASR + Gemini (best-effort)
            try:
                self.asr = LocalASR("models/vosk_en")
                print("[BOOT] ASR ready (Vosk)")
            except Exception as e:
                print("[WARN] ASR not ready:", e)
                self.asr = None

        except Exception:
            traceback.print_exc()
            self._add_msg("Setup problem. Please check model files.", is_user=False)

    def _feat_mfcc(self, y, sr):
        target=FEATURE_SETTINGS.get("sample_rate",16000)
        if sr!=target:
            y=librosa.resample(y, orig_sr=sr, target_sr=target); sr=target
        try:
            yt,_=librosa.effects.trim(y, top_db=30)
            if len(yt)>int(0.25*sr): y=yt
        except: pass
        return extract_mfcc(array=y, sr=sr)

    def _feat_ssl(self, y, sr):
        if sr!=16000:
            y=librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
        try:
            yt,_=librosa.effects.trim(y, top_db=30)
            if len(yt)>int(0.25*sr): y=yt
        except: pass
        return self.ssl(y)

    def _decode(self, probs):
        try:
            idx_h=self.classes.index("happy"); idx_s=self.classes.index("sad")
        except ValueError:
            idx_h,idx_s=0,1
        p_h,p_s=float(probs[idx_h]), float(probs[idx_s])
        p_max=max(p_h,p_s)
        if p_max < float(self.calib.get("min_confidence", 0.50)):
            return "Uncertain"
        if self.calib.get("mode","threshold")=="threshold":
            return "sad" if p_s >= float(self.calib.get("sad_threshold",0.57)) else "happy"
        return "happy" if p_h>=p_s else "sad"

    def _finish(self):
        self.mic_icon.setPixmap(self.mic_off)
        self.button.setText("🎤  Record & Detect"); self.button.setIcon(QIcon("mic-off.png"))
        self.button.setEnabled(True); self.is_processing=False

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
            # -------- 1) Record fixed 3s audio --------
            sr = FEATURE_SETTINGS.get("sample_rate", 16000)
            dur = float(self.calib.get("record_seconds", 3.0))
            y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            y = y.flatten()

            # Save last capture for debugging
            wavwrite("debug_last.wav", sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            print("[DEBUG] wrote debug_last.wav")

            # -------- 2) Basic audibility gate --------
            amp = float(np.max(np.abs(y)))
            print(f"[REC] amp={amp:.3f}, dur={dur:.2f}s, sr={sr}")
            if amp < AMP_MIN:
                msg = "I couldn't hear clearly. Please try a little closer to the mic."
                self._add_msg(msg, is_user=False)
                _speak_async(msg, self._finish)
                return

            intervals = librosa.effects.split(y, top_db=SPLIT_TOP_DB)
            nonsilent_sec = (np.sum([(j - i) for (i, j) in intervals]) / float(sr)) if len(intervals) else 0.0
            if nonsilent_sec < NON_SILENT_MIN_SEC:
                msg = "I didn’t catch that. Could you speak a bit longer and try again?"
                self._add_msg(msg, is_user=False)
                _speak_async(msg, self._finish)
                return

            # Optional quick user bubble showing that we captured audio
            # (We’ll still add the real transcript below)
            self._add_msg("🎤 (audio captured)", is_user=True)
            QApplication.processEvents()

            # -------- 3) Emotion inference (ALWAYS runs) --------
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")
            logits = self.session.run(None, {self.input_name: x})[0][0]
            probs = softmax(logits / max(1e-6, float(self.temperature)))
            label = self._decode(probs)  # "happy", "sad", or "Uncertain"
            print(f"[EMO] {label}")

            # -------- 4) Transcribe (if ASR available) --------
            user_text = ""
            if self.asr is not None:
                user_text = self.asr.transcribe(y, sr) or ""
                print(f"[ASR] {user_text!r}")

                if USE_ASR_GATE:
                    if len(user_text.split()) < ASR_MIN_WORDS:
                        msg = "I didn’t catch that clearly. Could you try once more?"
                        self._add_msg(msg, is_user=False)
                        _speak_async(msg, self._finish)
                        return
                else:
                    if len(user_text.strip()) == 0:
                        msg = "I didn’t hear any speech. When you’re ready, try sharing one sentence."
                        self._add_msg(msg, is_user=False)
                        _speak_async(msg, self._finish)
                        return

                # Show the transcript as the real user bubble
                if user_text:
                    self._add_msg(user_text, is_user=True)
                    self.history.add_user(user_text)
            else:
                # No ASR available: still move forward with a neutral cue
                self.history.add_user("(User spoke; transcript unavailable.)")

            # -------- 5) Tone hint for LLM (style only) --------
            system_prompt = (
                "You are a brief, warm companion. Keep replies to 1–2 short sentences, "
                "validate feelings, and ask at most one gentle follow-up. Avoid medical advice."
            )
            emotion_hint = (
                "Detected emotion: sad. Be validating and gentle." if label == "sad" else
                "Detected emotion: happy. Acknowledge positivity." if label == "happy" else
                "Detected emotion: uncertain. Ask for a simple clarification."
            )

            # -------- 6) Generate reply (fallback safe) --------
            if self.llm is not None:
                try:
                    bot_text = self.llm.generate(system_prompt, emotion_hint, self.history, max_tokens=200)
                    if not bot_text:
                        bot_text = RESPONSES.get(label, "I'm here with you.")
                except Exception:
                    traceback.print_exc()
                    bot_text = RESPONSES.get(label, "I'm here with you.")
            else:
                bot_text = RESPONSES.get(label, "I'm here with you.")

            self._add_msg(bot_text, is_user=False)
            self.history.add_assistant(bot_text)
            _speak_async(bot_text, self._finish)
            print(f"[STATE] asr={'yes' if self.asr else 'no'}, llm={'yes' if self.llm else 'no'}")

        except Exception:
            traceback.print_exc()
            self._add_msg("Something went wrong. Let's try again.", is_user=False)
            self._finish()

# ---- Run ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = EmotionApp(); w.show()
    sys.exit(app.exec_())
