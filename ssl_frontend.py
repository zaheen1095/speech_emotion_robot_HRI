# ssl_frontend.py  (single source of truth)
import os
# Prevent transformers from pulling torchvision paths
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from typing import Union
import numpy as np
import torch

# Import model-specific audio classes (avoids AutoProcessor vision paths)
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, AutoConfig

# Map your CLI friendly names to HF checkpoints + which class to use
_NAME_TO_HF = {
    "wav2vec2-base": ("facebook/wav2vec2-base", "wav2vec2"),
    "wav2vec2-base-960h": ("facebook/wav2vec2-base-960h", "wav2vec2"),
    "hubert-base": ("facebook/hubert-base-ls960", "hubert"),
}

class SSLFrontend:
    """
    Extracts frame-level SSL embeddings as float32 numpy of shape (T, D).
    Usage:
        ssl = SSLFrontend(model_name="wav2vec2-base", freeze=True, device="cpu")
        feats = ssl(y_16k_float_mono)  # -> (T, D)
    """
    def __init__(self, model_name: str = "wav2vec2-base",
                 freeze: bool = True,
                 device: Union[str, torch.device] = "cpu"):
        if model_name in _NAME_TO_HF:
            self.ckpt, kind = _NAME_TO_HF[model_name]
        else:
            # allow passing a raw HF id; default assume wav2vec2-compatible
            self.ckpt, kind = model_name, "wav2vec2"

        self.device = torch.device(device)

        # Load processor for Wav2Vec2; HuBERT uses the same processor interface
        # (Wav2Vec2Processor handles feature extractor + tokenizer bits we need)
        self.processor = Wav2Vec2Processor.from_pretrained(self.ckpt)

        if kind == "wav2vec2":
            self.model = Wav2Vec2Model.from_pretrained(self.ckpt)
        elif kind == "hubert":
            # HuBERT uses HubertModel
            self.model = HubertModel.from_pretrained(self.ckpt)
        else:
            # Fallback: try to inspect config and pick
            cfg = AutoConfig.from_pretrained(self.ckpt)
            if "hubert" in cfg.model_type:
                self.model = HubertModel.from_pretrained(self.ckpt)
            else:
                self.model = Wav2Vec2Model.from_pretrained(self.ckpt)

        self.model.to(self.device)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, y_16k: np.ndarray) -> np.ndarray:
        """
        y_16k: mono float32 waveform at 16 kHz
        returns last_hidden_state as (T, D) float32 numpy
        """
        # Ensure correct dtype/shape
        if not isinstance(y_16k, np.ndarray):
            y_16k = np.asarray(y_16k, dtype=np.float32)
        y_16k = y_16k.astype(np.float32, copy=False)

        inputs = self.processor(y_16k, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)                    # last_hidden_state: (1, T, D)
        feats = out.last_hidden_state.squeeze(0).detach().cpu().float().numpy()
        return feats
