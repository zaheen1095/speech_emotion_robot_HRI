# Speech Emotion Recognition Component (Happy vs Sad) — CPU / ONNX

This repository contains a **speech emotion recognition (SER) software component** developed for a robot interaction setup.
The classifier predicts **happy vs sad** from short English speech clips. The trained model can be exported to **ONNX** and used for **CPU-only inference** (e.g., in a Pepper-based lab demo).

**Important scope note:** This system predicts only a speech-based emotion label (**happy/sad**) and is used to adjust the robot’s reply style. It is **not** a clinical tool and does **not** infer mental health state.

---

## Data (not included)

Audio datasets are **not distributed** in this repository due to licensing restrictions.
To reproduce experiments, datasets must be downloaded from their official sources and used under their licence terms.

This project uses **only** utterances labelled **happy** and **sad**.
All audio is resampled to **16 kHz mono** before feature extraction.

### Datasets used
- CREMA-D  
- RAVDESS  
- TESS  
- SAVEE  
- JL-Corpus  
- IEMOCAP  

Only **happy** and **sad** labels are retained. Other emotion classes are ignored to keep a consistent binary task.

---

## Repository structure

### Data layout (example)
```text
datasets/
├── raw_audio/             # downloaded datasets (not included)
│   ├── train/
│   │   ├── happy/
│   │   └── sad/
│   └── test/
│       ├── happy/
│       └── sad/
├── resampled_audio/       # created by resample_audio.py (16 kHz mono)
├── augmented_audio/       # created by offline_augmentation.py (optional, train only)
├── features/              # MFCC features (.npy) from extract_features.py
└── features_ssl/          # cached SSL features from cache_ssl_features.py
```

If a different layout is used (for example, dataset-wise folders), update the paths in `config.py`.


## Repository structure (code and outputs)
```
models/
├── mfcc_v1/               # MFCC track outputs (checkpoints, ONNX, configs)
├── ssl_v1/                # SSL track outputs (checkpoints, ONNX, configs)
├── vosk_en/               # optional ASR resources (if used locally)
└── best_model_*.pt        # saved checkpoints from experiments (may be many)
```
# scripts (repo root)
resample_audio.py
offline_augmentation.py
extract_features.py
cache_ssl_features.py
train_model.py
test_model.py
export_to_onnx.py
onnx_inference.py
gui_live_predict.py        # GUI runtime (if used)
app.py                     # optional UI entry (if used)

> Submission note: keep `datasets/`, `models/`, and large `results/` outputs in `.gitignore` to keep the repo small.

## Installation

```bash
pip install -r requirements.txt
```

## Workflow (training → evaluation → deployment)
The pipeline is run in this order:

1. **Resample audio to 16 kHz mono** (required)  
```bash
python resample_audio.py
```

2. **Offline augmentation**
Creates augmented waveforms for the training split only.
```bash
python offline_augmentation.py
```
3. **Feature extraction (required)**
MFCC track:
```bash
python extract_features.py
```
SSL track:
```bash
 python cache_ssl_features.py
 ```
4. **Train model**
```bash 
python train_model.py
```

5. **Test model**
```bash
python test_model.py
```
6. **Export to ONNX**
```bash
python export_to_onnx.py
```

7. **Run inference (optional)**
```bash
python onnx_inference.py
```
8. **Application run (GUI)**
```bash
python gui_live_predict.py 
or 
python app.py
```

## Component behaviour

### Output labels
The SER model predicts:
- `happy`
- `sad`

In addition, the runtime may return:
- `uncertain`

`uncertain` is **not** trained as a third class. It is generated at runtime when:
- the input audio fails basic quality checks (e.g., too quiet / mostly silence), or
- the model confidence is below a configured threshold.

This prevents the system from forcing a label on weak or unreliable input.

---

## Integration guide (use this component in another system)

This repository is designed to work as a **loosely coupled SER component** that can be integrated into a larger robot or chatbot system.

### Expected input
- short speech clip
- 16 kHz, mono waveform

### Output format
For each input turn, the component returns:
- `label`: `"happy"`, `"sad"`, or `"uncertain"`
- probability scores (e.g., `p_happy`, `p_sad`)
- optional: confidence score or reason for uncertainty

### Recommended method (ONNX)
1. Load the exported ONNX model using ONNX Runtime.
2. For each user turn:
   - record audio
   - apply the same preprocessing as training
   - run ONNX inference
   - apply the decision rule (argmax or threshold + confidence gate)
3. Use the returned label to guide response behaviour in the host system.

See `onnx_inference.py` for a reference implementation.

### Alternative integration options
- **Subprocess call:** run `onnx_inference.py` from another language and parse output.
- **Python import:** import inference functions directly if the host system is Python-based.

---

## Deployment note (Pepper / robot setup)

Pepper (or another robot) can be used as the interaction interface (microphone + speech output), while SER inference runs on a host computer using the exported ONNX model.

- Dataset audio and trained checkpoints are not distributed in this repo.
- Use `onnx_inference.py` (or the GUI scripts) as a reference for running inference on recorded audio.

---

© 2026 – Thesis Project by Zaheen Fatima

---
