# Speech Emotion Recognition for Mental Health Robot

This project trains a speech emotion classifier (Happy/Sad) using MFCC features and integrates it into a robot (Pepper or IROBI).

##  Folder Structure
```
datasets/
├── raw_audio/           ← Original datasets
├── resampled_audio/     ← After 16kHz resampling
└── features/            ← Extracted MFCC .npy files

models/
├── cnn_bilstm.py        ← Model architecture
├── best_model.pt        ← Saved PyTorch model
└── model.onnx           ← ONNX model for robot
```

##  Workflow
1. `resample_audio.py` – Downsamples raw audio
2. `extract_features.py` – Extracts MFCC + delta
3. `train_model.py` – Trains CNN + BiLSTM
4. `test_model.py` – Evaluates accuracy
5. `export_to_onnx.py` – Exports to ONNX for Pepper/IROBI
6. `live_inference.py` – Real-time emotion via mic

##  Installation
```bash
pip install -r requirements.txt
```

##  Deployment
Use `onnx_inference.py` on robot platform with audio input to recognize emotions.

---

© 2025 – Thesis Project by Zaheen Fatima
