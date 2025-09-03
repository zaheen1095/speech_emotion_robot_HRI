# ssl_frontend.py
from typing import Union
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel

_NAME_MAP = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-base-960h": "facebook/wav2vec2-base-960h",
    "hubert-base": "facebook/hubert-base-ls960"
}

class SSLFrontend:
    """
    Returns float32 numpy features of shape (T, D) = last_hidden_state frames.
    - freeze=True -> eval mode, gradients off
    - model_name may be a shortcut in _NAME_MAP or a full HF id
    """
    def __init__(self, model_name: str = "wav2vec2-base",
                 freeze: bool = True,
                 device: Union[str, torch.device] = "cpu"):
        self.id = _NAME_MAP.get(model_name, model_name)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(self.id)
        self.model = AutoModel.from_pretrained(self.id).to(self.device)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.model.eval()
        self.freeze = freeze

    @torch.inference_mode()
    def __call__(self, y_16k: np.ndarray) -> np.ndarray:
        # y_16k: mono float32 array at 16 kHz
        inputs = self.processor(y_16k, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # last_hidden_state: (1, T, D)
        feats = out.last_hidden_state.squeeze(0).detach().cpu().float().numpy()
        return feats
