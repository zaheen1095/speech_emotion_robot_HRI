# gui_live_predict.py
import sys, os, json, threading, traceback
import numpy as np
import librosa, sounddevice as sd, pyttsx3

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
from faster_whisper import WhisperModel
from scipy.io.wavfile import write as wavwrite

# ---- TUNABLES ----
USE_ASR_GATE = False                 # set False to disable word-count gate
AMP_MIN = 0.035                      # min peak amplitude to consider "audible"
NON_SILENT_MIN_SEC = 0.5             # easier threshold for short turns
SPLIT_TOP_DB = 30                    # silence splitter aggressiveness
ASR_MIN_WORDS = 2                    # allow short phrases
sd.default.samplerate = 16000
sd.default.channels = 1
FIRST_TURN_EMO_ONLY = False          # keep False for API-driven replies every turn

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES
from extract_features import extract_mfcc
# from ssl_frontend import SSLFrontend  # only if SSL model present

import onnxruntime as ort
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Dict

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

def _smoke_test_gemini(model_name: str) -> bool:
    try:
        import google.generativeai as genai
        print("[GEMINI] sdk_version:", getattr(genai, "__version__", "unknown"))
        gm = genai.GenerativeModel(model_name)
        r = gm.generate_content("Say: PONG")
        print("[GEMINI] smoke:", (r.text or "").strip())
        return True
    except Exception as e:
        print("[GEMINI] smoke FAILED:", repr(e))
        return False

def _pick_gemini_model(api_key: str) -> str:
    # Configure first so list_models works
    genai.configure(api_key=api_key)
    try:
        models = list(genai.list_models())
        names = [
            getattr(m, "name", "")
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        # Prefer stable flash models
        for pref in ("models/gemini-2.5-flash", "models/gemini-2.0-flash",
                     "gemini-flash-latest", "models/gemini-pro-latest"):
            if pref in names:
                print("[GEMINI] selected:", pref)
                return pref
        if names:
            print("[GEMINI] selected (first):", names[0])
            return names[0]
    except Exception as e:
        print("[GEMINI] list_models failed after configure:", repr(e))
    return "models/gemini-pro-latest"

# ---- Faster-Whisper ASR ----
class LocalASR:
    def __init__(self, model_dir=None):
        # "base.en" is a good balance; use "tiny.en" for very slow CPUs
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")

    def transcribe(self, y: np.ndarray, sr: int) -> str:
        # 1) mono 16k
        if y.ndim > 1:
            y = y[:,0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        # 2) normalize only (no hard gates)
        y = librosa.util.normalize(y.astype(np.float32), axis=0)
        # 3) transcribe with VAD
        segments, _ = self.model.transcribe(
            y, language="en", vad_filter=True, vad_parameters={"min_silence_duration_ms": 200}
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return " ".join(text.split())

# ---- Minimal chat memory ----
@dataclass
class ChatHistory:
    messages: List[Dict[str, str]] = field(default_factory=list)
    def add_user(self, t: str): self.messages.append({"role": "user", "content": t})
    def add_assistant(self, t: str): self.messages.append({"role": "assistant", "content": t})
    def last_user(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m.get("content","")
        return ""
    def as_prompt(self, last: int = 3) -> str:  # keep context tiny to avoid token limit
        msgs = self.messages[-last:] if last else self.messages
        if not msgs: return "Assistant:"
        return "\n".join(
            ("User: " + m["content"]) if m["role"] == "user" else ("Assistant: " + m["content"])
            for m in msgs
        ) + "\nAssistant:"

# ---- Gemini helpers ----
def _resp_text_and_finish(resp):
    text = ""
    finish = None
    # Try quick accessor
    try:
        text = (getattr(resp, "text", "") or "").strip()
    except Exception:
        text = ""
    # Fallback to parts
    if not text and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        finish = getattr(cand, "finish_reason", None)
        parts = getattr(getattr(cand, "content", None), "parts", []) or []
        text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None)).strip()
    return text, finish

class GeminiClient:
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise RuntimeError("Gemini API key missing. Add it to secrets.json.")
        genai.configure(api_key=api_key)
        self.model_name = model

    # returns (text, finish_reason or None)
    def generate(self, system_prompt: str, emotion_hint: str,
                 history: ChatHistory, max_tokens: int = 180):
        try:
            gm = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=(
                    f"Be warm and brief (<=30 words). One gentle question at most. "
                    f"{emotion_hint}"
                )
            )
            resp = gm.generate_content(
                history.as_prompt(last=3),  # small context prevents MAX_TOKENS
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.4,
                    "candidate_count": 1,
                }
            )
            text, finish = _resp_text_and_finish(resp)
            print(f"[LLM] text={bool(text)} finish={finish!r}")
            return text, finish
        except Exception as e:
            print("[LLM] EXC:", repr(e))
            return "", None

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
                picked = _pick_gemini_model(api_key)
                self.llm = GeminiClient(model=picked, api_key=api_key)
                print("[BOOT] Gemini client ready", picked)
                _smoke_test_gemini(picked)
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
            self.calib["record_seconds"] = max(4.5, float(self.calib.get("record_seconds", 3.0)))
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

            # ASR
            try:
                self.asr = LocalASR()
                print("[BOOT] ASR ready (Faster-Whisper base.en int8)")
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
            # -------- 1) Record fixed-duration audio --------
            sr = FEATURE_SETTINGS.get("sample_rate", 16000)
            dur = float(self.calib.get("record_seconds", 3.0))
            y = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            y = y.flatten()

            # Save last capture for debugging
            try:
                wavwrite("debug_last.wav", sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
                print("[DEBUG] wrote debug_last.wav")
            except Exception:
                traceback.print_exc()

            # -------- 2) Audibility + non-silent check --------
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

            # quick visual tick
            self._add_msg("🎤 (audio captured)", is_user=True)
            QApplication.processEvents()

            # -------- 3) Emotion inference --------
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")
            logits = self.session.run(None, {self.input_name: x})[0][0]
            probs = softmax(logits / max(1e-6, float(self.temperature)))
            label = self._decode(probs)
            print(f"[EMO] {label}")

            # -------- 4) Transcribe (no hard return on empty) --------
            user_text = ""
            if self.asr is not None:
                user_text = self.asr.transcribe(y, sr) or ""
                print(f"[ASR] {user_text!r}")

                if USE_ASR_GATE and len(user_text.split()) < ASR_MIN_WORDS and user_text.strip():
                    self._add_msg("Could you say that one more time in a sentence?", is_user=False)

                if user_text.strip():
                    self._add_msg(user_text, is_user=True)
                    self.history.add_user(user_text)
                else:
                    # NO EARLY RETURN — add neutral marker and still go to LLM
                    self._add_msg("I didn’t catch words—try one short sentence?", is_user=False)
                    self.history.add_user("(User spoke; transcript unavailable.)")
            else:
                self.history.add_user("(User spoke; transcript unavailable.)")

            # -------- 5) Tone hint for LLM --------
            system_prompt = (
                "You are a brief, warm companion. Keep replies to 1–2 short sentences, "
                "validate feelings, and ask at most one gentle follow-up. Avoid medical advice."
            )
            emotion_hint = (
                "Detected emotion: sad. Be validating and gentle." if label == "sad" else
                "Detected emotion: happy. Acknowledge positivity." if label == "happy" else
                "Detected emotion: uncertain. Ask for a simple clarification."
            )

            # -------- 6) Generate reply (no duplicate prefixes) --------
            bot_text, finish = "", None
            if self.llm is not None:
                bot_text, finish = self.llm.generate(system_prompt, emotion_hint, self.history, max_tokens=180)

            if not bot_text:
                # true minimal fallback if LLM returns empty
                emotion_short = "sad" if label == "sad" else "happy" if label == "happy" else "uncertain"
                bot_text = _supportive_fallback(user_text, emotion_short)
            else:
                # add a single, lightweight cue ONCE
                cue = {"sad": "I’m hearing sadness. ",
                       "happy": "You sound positive. "}.get(label, "")
                bot_text = cue + bot_text

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
