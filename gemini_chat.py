# gui_live_predict.py
# Requires: pip install webrtcvad google-generativeai onnxruntime librosa sounddevice pyttsx3
# Put a Vosk model (e.g., vosk-model-en-us-0.22) at models/vosk_en/
import sys, os, json, threading, traceback, time
import numpy as np
import librosa, sounddevice as sd, pyttsx3, webrtcvad

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame

# ========= TUNABLES (stable + forgiving) =========
FIRST_TURN_EMO_ONLY   = True   # First press: speak 1 supportive line from emotion, then end that turn
USE_ASR_GATE          = True   # If transcript is very short, show a gentle nudge (but DON'T block LLM)
ASR_MIN_WORDS         = 2
AMP_MIN               = 0.05   # minimum peak amplitude to accept audio
SAVE_DEBUG_WAV        = True
sd.default.samplerate = 16000
sd.default.channels   = 1

# WebRTC VAD (natural utterance capture)
VAD_FRAME_MS          = 30     # 10/20/30 ms
VAD_AGGRESSIVENESS    = 2      # 0..3
VAD_MAX_SEC           = 12.0   # safety cap
VAD_TRAIL_SIL_MS      = 700    # stop after trailing silence
# =================================================

# --- Project modules ---
from config import FEATURE_SETTINGS, RESPONSES
from extract_features import extract_mfcc
# from ssl_frontend import SSLFrontend  # only if you use the SSL frontend

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
        if os.path.exists(p): return p
    return None

def _load_secret_key(path="secrets.json"):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        full = os.path.join(base, path)
        with open(full, "r", encoding="utf-8") as f:
            key = json.load(f).get("GEMINI_API_KEY", "").strip()
            return key
    except Exception:
        return ""

def _supportive_fallback(user_text: str, emotion: str) -> str:
    s = (user_text or "").strip()
    if len(s) > 80: s = s[:80].rstrip() + "…"
    if emotion == "sad":   return f"I hear you about “{s}”. That sounds heavy. What part feels hardest right now?"
    if emotion == "happy": return f"That’s lovely to hear about “{s}”. What made it feel so good?"
    return "I want to make sure I understood. Could you say that again in a few words?"

# ========= WebRTC VAD recorder =========
class VADRecorder:
    """
    Capture natural-length utterances with WebRTC VAD at 16 kHz mono.
    Stops after VAD_TRAIL_SIL_MS of trailing silence or VAD_MAX_SEC.
    """
    def __init__(self, sr=16000, frame_ms=30, aggressiveness=2, max_sec=12.0, trailing_sil_ms=700):
        assert frame_ms in (10, 20, 30), "webrtcvad supports only 10, 20, or 30 ms"
        self.sr = sr
        self.frame_ms = frame_ms
        self.frame_len = int(sr * (frame_ms / 1000.0))
        self.vad = webrtcvad.Vad(aggressiveness)
        self.max_frames = int(max_sec * 1000 / frame_ms)
        self.trail_needed = int(trailing_sil_ms / frame_ms)

    def _bytes_from_float32(self, y):
        pcm16 = (np.clip(y, -1, 1) * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def _float32_from_bytes(self, b):
        pcm16 = np.frombuffer(b, dtype=np.int16)
        return (pcm16.astype(np.float32) / 32767.0)

    def capture(self):
        sr = self.sr; ch = 1; block = self.frame_len
        buf = bytearray(); sil_in_row = 0; frames = 0

        stream = sd.InputStream(samplerate=sr, channels=ch, dtype='float32', blocksize=block)
        with stream:
            while True:
                chunk, _ = stream.read(block)    # shape: (block, 1)
                y = chunk[:, 0]
                b = self._bytes_from_float32(y)
                is_speech = self.vad.is_speech(b, sr)

                buf.extend(b)
                frames += 1
                sil_in_row = 0 if is_speech else (sil_in_row + 1)

                if sil_in_row >= self.trail_needed:  # trailing silence → stop
                    break
                if frames >= self.max_frames:        # safety cap
                    break

        y = self._float32_from_bytes(bytes(buf))
        # small pad so last syllable isn't clipped
        pad = int(0.1 * sr)
        if pad > 0:
            y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
        return y, sr

# ---- Vosk ASR (soft preprocessing; no hard blocks) ----
class LocalASR:
    def __init__(self, model_dir="models/vosk_en"):
        from vosk import Model
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Vosk model dir missing: {model_dir}")
        self.model = Model(model_dir)

    def transcribe(self, y: np.ndarray, sr: int) -> str:
        from vosk import KaldiRecognizer
        import json as pyjson
        if y.ndim > 1: y = y[:,0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
        try:
            yt, _ = librosa.effects.trim(y, top_db=25)
            if len(yt) > 0.25 * sr: y = yt
        except Exception:
            pass
        y = librosa.util.normalize(y, axis=0)

        rec = KaldiRecognizer(self.model, sr)
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
        step = sr // 2  # ~0.5s chunks
        for i in range(0, len(pcm), step*2):
            rec.AcceptWaveform(pcm[i:i+step*2])
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

# ---- Gemini helpers ----
_PREFERRED_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

def _try_model_name(gm_name):
    """Return a list of name variants to try for a given model."""
    return [gm_name, f"models/{gm_name}"]

def _configure_genai_from_secrets():
    key = _load_secret_key()
    if not key:
        raise RuntimeError("Gemini API key missing in secrets.json")
    # also set env var so list_models() works consistently
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)

def _pick_gemini_model():
    """Pick an available model; if listing fails, use a preferred fallback."""
    try:
        models = list(genai.list_models())
        gens = [m for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        names = [getattr(m, "name", "") for m in gens]
        print("[GEMINI] generate-capable models:"); [print("  -", n) for n in names]
        # Return first preferred that appears (in plain or prefixed form)
        for want in _PREFERRED_MODELS:
            for n in names:
                if n.endswith(want) or n == want:
                    print("[GEMINI] selected:", n)
                    return n
        if names:
            print("[GEMINI] selected (first):", names[0])
            return names[0]
    except Exception as e:
        print("[GEMINI] list_models failed:", repr(e))
    # Fallback if listing fails
    return _PREFERRED_MODELS[0]

class GeminiClient:
    def __init__(self, model: str):
        self.model_name = model

    def generate(self, system_prompt: str, emotion_hint: str, history: ChatHistory, max_tokens: int=220) -> str:
        """Robust generate: tries name variants, handles empty/safety responses."""
        prompt = history.as_prompt()
        sys_instr = (
            system_prompt + "\n" + emotion_hint +
            "\nKeep replies supportive and concrete. Validate feelings. "
            "Ask at most one gentle follow-up. Avoid medical advice."
        )

        # Try model name variants to avoid NotFound v1beta quirks
        for candidate_name in _try_model_name(self.model_name):
            try:
                gm = genai.GenerativeModel(model_name=candidate_name, system_instruction=sys_instr)
                resp = gm.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})

                # Prefer resp.text if present; otherwise stitch parts safely
                text = (getattr(resp, "text", "") or "").strip()
                if not text and getattr(resp, "candidates", None):
                    cand = resp.candidates[0]
                    # If safety blocked -> finish_reason may be 2; guard for missing parts
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", []) if content else []
                    text = "".join(getattr(p, "text", "") for p in parts).strip()

                if text:
                    print(f"[LLM] ok via {candidate_name} (len={len(text)})")
                    return text
                else:
                    print(f"[LLM] empty via {candidate_name} (likely safety).")
            except Exception as e:
                print(f"[LLM] EXC via {candidate_name}:", repr(e))

        return ""  # caller will fallback

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
        self.setGeometry(100, 100, 400, 520)

        self.mic_icon = QLabel()
        self.mic_off = QPixmap("mic-off.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mic_on  = QPixmap("mic-on.png").scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mic_icon.setPixmap(self.mic_off)

        self.chat_layout = QVBoxLayout(); self.chat_layout.setAlignment(Qt.AlignTop)
        container = QWidget(); container.setLayout(self.chat_layout)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container); self.scroll = scroll

        self.button = QPushButton("🎤  Record / Talk")
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

        # VAD capture
        self.vadrec = VADRecorder(
            sr=16000,
            frame_ms=VAD_FRAME_MS,
            aggressiveness=VAD_AGGRESSIVENESS,
            max_sec=VAD_MAX_SEC,
            trailing_sil_ms=VAD_TRAIL_SIL_MS,
        )

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
                _configure_genai_from_secrets()
                picked = _pick_gemini_model()
                # If the name is prefixed "models/...", normalize to bare name for our retry loop
                for want in _PREFERRED_MODELS:
                    if picked.endswith(want):
                        picked = want
                        break
                self.llm = GeminiClient(model=picked)
                print("[BOOT] Gemini client ready", picked)
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

            # SSL frontend if needed
            if self.model_type=="ssl" and self.ssl is None:
                try:
                    import warnings
                    from transformers.utils import logging as hf_logging
                    hf_logging.set_verbosity_error()
                    warnings.filterwarnings("ignore", "Passing `gradient_checkpointing`.*")
                    warnings.filterwarnings("ignore", "`clean_up_tokenization_spaces`.*")
                    from ssl_frontend import SSLFrontend
                    self.ssl=SSLFrontend()
                except Exception as e:
                    print("[WARN] SSL frontend not ready:", e)

            # ASR
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
        except Exception:
            pass
        return extract_mfcc(array=y, sr=sr)

    def _feat_ssl(self, y, sr):
        if sr!=16000:
            y=librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
        try:
            yt,_=librosa.effects.trim(y, top_db=30)
            if len(yt)>int(0.25*sr): y=yt
        except Exception:
            pass
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
        self.button.setText("🎤  Record / Talk"); self.button.setIcon(QIcon("mic-off.png"))
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
            # ===== VAD-based capture (natural utterance) =====
            y, sr = self.vadrec.capture()
            if SAVE_DEBUG_WAV:
                try:
                    wavwrite("debug_last.wav", sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
                    print("[DEBUG] wrote debug_last.wav")
                except Exception:
                    traceback.print_exc()

            amp = float(np.max(np.abs(y))); secs = len(y)/float(sr)
            print(f"[REC] amp={amp:.3f}, secs={secs:.2f}, sr={sr}")
            if amp < AMP_MIN:
                msg = "I couldn't hear clearly. Please try a little closer to the mic."
                self._add_msg(msg, is_user=False)
                _speak_async(msg, self._finish)
                return

            # quick tick that we captured (we'll still add transcript bubble later)
            self._add_msg("🎤 (audio captured)", is_user=True)
            QApplication.processEvents()

            # ===== Emotion inference =====
            feats = self._feat_ssl(y, sr) if self.model_type == "ssl" else self._feat_mfcc(y, sr)
            x = feats[np.newaxis, :, :].astype("float32")
            logits = self.session.run(None, {self.input_name: x})[0][0]
            probs = softmax(logits / max(1e-6, float(self.temperature)))
            label = self._decode(probs)
            emo_conf = float(np.max(probs))
            print(f"[EMO] {label} (conf={emo_conf:.2f})")

            # ===== First turn: emotion-only (then return) =====
            if FIRST_TURN_EMO_ONLY and not self.has_started_chat:
                reply = RESPONSES.get(label, "I'm here with you.")
                self._add_msg(reply, is_user=False)
                self.has_started_chat = True
                _speak_async(reply, self._finish)
                return

            # ===== Transcribe (soft gate) =====
            user_text = ""
            if self.asr is not None:
                user_text = self.asr.transcribe(y, sr) or ""
                print(f"[ASR] {user_text!r}")

                if not user_text.strip():
                    nudge = "I didn’t catch that—could you say it again in a sentence?"
                    self._add_msg(nudge, is_user=False)
                    # Still add a neutral user marker so LLM can continue if needed
                    self.history.add_user("(User spoke; transcript unavailable.)")
                else:
                    if USE_ASR_GATE and len(user_text.split()) < ASR_MIN_WORDS:
                        self._add_msg("Could you say that one more time in a sentence?", is_user=False)
                    self._add_msg(user_text, is_user=True)
                    self.history.add_user(user_text)
            else:
                self.history.add_user("(User spoke; transcript unavailable.)")

            # ===== LLM tone prompt =====
            system_prompt = (
                "You are a brief, warm companion. Keep replies to 1–2 short sentences, "
                "validate feelings, and ask at most one gentle follow-up. Avoid medical advice."
            )
            emotion_hint = (
                "Detected emotion: sad. Be validating and gentle." if label == "sad" else
                "Detected emotion: happy. Acknowledge positivity." if label == "happy" else
                "Detected emotion: uncertain. Ask for a simple clarification."
            )

            # ===== Generate reply (robust) =====
            if self.llm is not None:
                bot_text = self.llm.generate(system_prompt, emotion_hint, self.history, max_tokens=200)
                if not bot_text:
                    bot_text = _supportive_fallback(user_text, "sad" if label=="sad" else "happy" if label=="happy" else "uncertain")
            else:
                bot_text = _supportive_fallback(user_text, "sad" if label=="sad" else "happy" if label=="happy" else "uncertain")

            self._add_msg(bot_text, is_user=False)
            self.history.add_assistant(bot_text)
            _speak_async(bot_text, self._finish)

        except Exception:
            traceback.print_exc()
            self._add_msg("Something went wrong. Let's try again.", is_user=False)
            self._finish()

# ---- Run ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = EmotionApp(); w.show()
    sys.exit(app.exec_())
