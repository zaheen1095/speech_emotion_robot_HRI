import torch, numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor

_NAME_TO_HF = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "hubert-base":   "facebook/hubert-base-ls960"
}

class SSLFrontend:
    def __init__(self, model_name="wav2vec2-base", freeze=True, device="cpu"):
        self.proc = Wav2Vec2Processor.from_pretrained(_NAME_TO_HF[model_name])
        self.model = Wav2Vec2Model.from_pretrained(_NAME_TO_HF[model_name])
        if freeze:
            for p in self.model.parameters(): p.requires_grad = False
        self.model.eval()
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, wav_16k: np.ndarray) -> np.ndarray:
        # wav_16k: mono float32, 16 kHz
        inputs = self.proc(wav_16k, sampling_rate=16000, return_tensors="pt", padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs).last_hidden_state.squeeze(0)  # [T, D]
        return out.cpu().numpy()
