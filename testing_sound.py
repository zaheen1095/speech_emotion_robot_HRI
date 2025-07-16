# eval_test.py
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from extract_features import extract_mfcc
from config import FEATURE_SETTINGS, CLASSES
from models.cnn_bilstm import CNNBiLSTM

# 1) load your model
model = CNNBiLSTM(
    input_dim=FEATURE_SETTINGS['n_mfcc'] * (
        1 + int(FEATURE_SETTINGS['use_delta']) + int(FEATURE_SETTINGS['use_delta_delta'])
    ),
    num_classes=len(CLASSES)
)
ckpt = torch.load("models/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# 2) walk through every .wav under datasets/raw_audio/test/{happy,sad}
y_true, y_pred = [], []
ROOT = Path("datasets/raw_audio/test")
for label in CLASSES:
    for wav in (ROOT/label).glob("*.wav"):
        feats = extract_mfcc(audio_path=str(wav))
        inp   = torch.tensor(feats[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            logits = model(inp)
            probs  = torch.softmax(logits, dim=1)[0].numpy()
        pred = CLASSES[int(np.argmax(probs))]
        y_true.append(label)
        y_pred.append(pred)

# 3) print report
print(classification_report(y_true, y_pred, target_names=CLASSES))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=CLASSES))
