# ser_component.py
# A small wrapper to integrate the SER model (happy vs sad) into any other software.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort

# Optional deps (only needed for convenience helpers)
try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
except Exception:
    librosa = None

# Project deps (these exist in your repo)
from config import FEATURE_SETTINGS
from extract_features import extract_mfcc
from ssl_frontend import SSLFrontend


@dataclass
class SEROutput:
    label: str                      # "happy" | "sad" | "uncertain"
    prob_happy: float
    prob_sad: float
    confidence: float               # max(prob_happy, prob_sad)
    used_thresholding: bool
    sad_threshold: float
    temperature: float
    model_type: str                 # "mfcc" | "ssl"
    details: Dict[str, Any]


class SERComponent:
    """
    Speech Emotion Recognition component wrapper (happy vs sad).

    Integration idea:
      - The chatbot / main software calls SERComponent.predict(y, sr)
      - It receives SEROutput (label + probabilities)
      - The chatbot uses label as a cue to choose response style.

    This class is intentionally "thin":
      - no UI
      - no Pepper networking
      - no ASR
      - just feature extraction + ONNX inference + decision rule
    """

    def __init__(
        self,
        models_root: Union[str, Path] = "models",
        prefer_ssl: bool = True,
        providers: Optional[list] = None,
        force_model_type: Optional[str] = None,  # "ssl" or "mfcc"
        use_gate: bool = True,
        min_amp: float = 0.02,
        default_record_sr: int = 16000,
    ):
        self.models_root = Path(models_root)
        self.prefer_ssl = bool(prefer_ssl)
        self.providers = providers or ["CPUExecutionProvider"]

        self.use_gate = bool(use_gate)
        self.min_amp = float(min_amp)
        self.default_record_sr = int(default_record_sr)

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None

        self.model_type: str = "mfcc"
        self.classes = ["happy", "sad"]

        # Calibration / decision rule
        self.temperature: float = 1.0
        self.sad_threshold: float = 0.50
        self.min_confidence: float = 0.0

        # SSL frontend (lazy-init)
        self._ssl: Optional[SSLFrontend] = None

        # Load model immediately
        self._auto_load_model(force_model_type=force_model_type)

    # ---------------------------
    # Public API
    # ---------------------------

    def predict(
        self,
        y: np.ndarray,
        sr: int,
        return_uncertain: bool = True,
    ) -> SEROutput:
        """
        Predict emotion from waveform.
        y: mono float waveform (np.ndarray)
        sr: sampling rate
        """
        if self.session is None or self.input_name is None:
            raise RuntimeError("SERComponent is not ready: ONNX session not loaded.")

        y = self._ensure_mono_float(y)

        if self.use_gate:
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak < self.min_amp:
                return SEROutput(
                    label="uncertain" if return_uncertain else "happy",
                    prob_happy=0.0,
                    prob_sad=0.0,
                    confidence=0.0,
                    used_thresholding=True,
                    sad_threshold=self.sad_threshold,
                    temperature=self.temperature,
                    model_type=self.model_type,
                    details={"reason": "too_quiet", "peak": peak},
                )

        # resample to pipeline SR if needed
        target_sr = int(FEATURE_SETTINGS.get("sample_rate", 16000))
        if sr != target_sr:
            if librosa is None:
                raise RuntimeError("librosa is required to resample audio.")
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        feats = self._extract_features(y, sr)  # (T, D)
        x = feats[np.newaxis, :, :].astype(np.float32)  # (1, T, D)

        logits = self.session.run(None, {self.input_name: x})[0][0]  # (2,)
        probs = self._softmax(logits, T=self.temperature)

        idx_h = self._safe_index("happy", default=0)
        idx_s = self._safe_index("sad", default=1)

        prob_h = float(probs[idx_h])
        prob_s = float(probs[idx_s])
        conf = float(max(prob_h, prob_s))

        # Decision rule:
        # - threshold on p(sad) for stability (matches repo behaviour)
        # - optional min_confidence -> "uncertain"
        used_thresholding = True
        if conf < float(self.min_confidence or 0.0) and return_uncertain:
            label = "uncertain"
        else:
            label = "sad" if prob_s >= float(self.sad_threshold) else "happy"

        return SEROutput(
            label=label,
            prob_happy=prob_h,
            prob_sad=prob_s,
            confidence=conf,
            used_thresholding=used_thresholding,
            sad_threshold=float(self.sad_threshold),
            temperature=float(self.temperature),
            model_type=self.model_type,
            details={
                "logits": [float(logits[0]), float(logits[1])],
                "classes": list(self.classes),
                "input_shape": list(x.shape),
            },
        )

    def predict_wav(self, wav_path: Union[str, Path], return_uncertain: bool = True) -> SEROutput:
        """
        Convenience helper if the integrator already has a wav file.
        """
        if sf is None:
            raise RuntimeError("soundfile is required for predict_wav(). Install: pip install soundfile")

        wav_path = Path(wav_path)
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y[:, 0]
        return self.predict(y=y, sr=int(sr), return_uncertain=return_uncertain)

    # ---------------------------
    # Internal: model loading
    # ---------------------------

    def _auto_load_model(self, force_model_type: Optional[str] = None) -> None:
        """
        Auto-pick SSL first (if prefer_ssl), else MFCC.
        Expects:
          models/ssl_v1/model_ssl.onnx (+ optional calibration.json)
          models/mfcc_v1/model_mfcc.onnx (+ optional calibration.json)
        """
        tracks = [
            {"type": "ssl", "dir": self.models_root / "ssl_v1", "files": ["model_ssl.onnx"]},
            {"type": "mfcc", "dir": self.models_root / "mfcc_v1", "files": ["model_mfcc.onnx", "model_mfcc_int8.onnx"]},
        ]

        if force_model_type in ("ssl", "mfcc"):
            tracks = [t for t in tracks if t["type"] == force_model_type]

        if not self.prefer_ssl:
            tracks = list(reversed(tracks))

        chosen = None
        for t in tracks:
            onnx_path = self._find_existing([t["dir"] / fn for fn in t["files"]])
            if onnx_path is not None:
                chosen = (t["type"], t["dir"], onnx_path)
                break

        if chosen is None:
            raise FileNotFoundError(
                f"No ONNX model found under {self.models_root}. "
                f"Expected models/ssl_v1/model_ssl.onnx or models/mfcc_v1/model_mfcc.onnx"
            )

        model_type, model_dir, onnx_path = chosen
        self.model_type = model_type

        # Load calibration if present (threshold/temperature/classes/min_confidence)
        self._load_calibration(model_dir)

        # Create ONNX session
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3

        self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

        # If SSL selected, lazy init frontend (only if needed)
        if self.model_type == "ssl":
            self._ssl = None  # lazy create on first call

    def _load_calibration(self, model_dir: Path) -> None:
        """
        Supported:
          calibration.json (preferred in repo for ONNX runtime)
        Typical fields used in repo calibration flow:
          - threshold / sad_threshold
          - temperature
          - min_confidence
          - classes
        """
        calib_path = model_dir / "calibration.json"
        if not calib_path.exists():
            return

        try:
            d = json.loads(calib_path.read_text(encoding="utf-8"))

            # accept both key names (repo scripts have slightly different outputs)
            if "classes" in d and isinstance(d["classes"], list) and len(d["classes"]) >= 2:
                self.classes = d["classes"][:2]

            if "sad_threshold" in d:
                self.sad_threshold = float(d["sad_threshold"])
            elif "threshold" in d:
                # make_calibration_from_onnx writes "threshold"
                self.sad_threshold = float(d["threshold"])

            if "temperature" in d:
                self.temperature = float(d["temperature"])

            if "min_confidence" in d:
                self.min_confidence = float(d["min_confidence"])

        except Exception:
            # keep defaults if calibration is malformed
            pass

    # ---------------------------
    # Internal: features
    # ---------------------------

    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        MFCC path uses extract_mfcc() from repo.
        SSL path uses SSLFrontend() from repo.
        """
        if self.model_type == "mfcc":
            return extract_mfcc(array=y, sr=sr)  # (T, D) :contentReference[oaicite:1]{index=1}

        # SSL
        if self._ssl is None:
            # Device selection is inside SSLFrontend; keep it simple (CPU is fine)
            self._ssl = SSLFrontend(
                model_name="wav2vec2-base",
                freeze=True,
                device="cpu",
            )  # :contentReference[oaicite:2]{index=2}
        return self._ssl(y)

    # ---------------------------
    # Utilities
    # ---------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
        x = logits.astype(np.float32)
        T = float(max(1e-6, T))
        x = x / T
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _safe_index(self, name: str, default: int) -> int:
        try:
            return self.classes.index(name)
        except Exception:
            return int(default)

    @staticmethod
    def _ensure_mono_float(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim > 1:
            y = y[:, 0]
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    @staticmethod
    def _find_existing(paths: list[Path]) -> Optional[Path]:
        for p in paths:
            if p.exists():
                return p
        return None

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="SERComponent CLI test")
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file")
    parser.add_argument("--models_root", type=str, default="models")
    parser.add_argument("--prefer_ssl", action="store_true")

    args = parser.parse_args()

    ser = SERComponent(
        models_root=args.models_root,
        prefer_ssl=args.prefer_ssl,
    )

    out = ser.predict_wav(args.wav)

    print(json.dumps({
        "label": out.label,
        "confidence": out.confidence,
        "prob_happy": out.prob_happy,
        "prob_sad": out.prob_sad,
        "model_type": out.model_type
    }, indent=2))