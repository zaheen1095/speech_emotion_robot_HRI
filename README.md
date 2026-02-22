# Speech Emotion Recognition for Mental Health Robot (Happy vs Sad)

This repository contains code for a speech emotion recognition (SER) component developed for a robot interaction setup.
The classifier predicts **happy vs sad** from short speech clips. The trained model can be exported to **ONNX** and used in a CPU runtime (for example, with Pepper).

---

## Data (not included in this repository)

Audio datasets are **not uploaded** in this repository due to licensing restrictions.
To reproduce the experiments, datasets must be downloaded from their official sources and used under their licence terms.

Only utterances labelled **happy** and **sad** are used. Other emotion classes are ignored to keep the task binary and consistent across corpora.
All audio was resampled to **16 kHz mono** before feature extraction to ensure consistency across corpora.

### Datasets used in this project

The experiments were conducted using the following English speech emotion corpora:

- CREMA-D
- RAVDESS
- TESS
- SAVEE
- JL-Corpus
- IEMOCAP

These datasets contain acted or semi-structured emotional speech recordings.
Only the **happy** and **sad** emotion labels were retained for this project.
All other emotion classes were removed to maintain a binary classification setup.

This repository does not redistribute any audio files.

### Expected local folder structure (matches this project)

```text
datasets/
├── raw_audio/             # downloaded datasets (not included in repo)
│   ├── train/
│   │   ├── happy/         # .wav files
│   │   └── sad/           # .wav files
│   └── test/
│       ├── happy/         # .wav files
│       └── sad/           # .wav files
├── resampled_audio/       # created by resample_audio.py (16 kHz)
├── augmented_audio/       # created by offline_augmentation.py (optional, train only)
├── features/              # MFCC feature .npy files (extract_features.py output)
└── features_ssl/          # SSL cached features (cache_ssl_features.py output)
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

1. **Resample audio** (required)  
Converts all audio to a consistent format (16 kHz, mono).
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

7. **Run inference(Optionl)**
```bash
python onnx_inference.py
```
8. **Application run (GUI)**
```bash
python gui_live_predict.py 
or 
python app.py
```

##  Deployment
Pepper (or another robot) is used as the interaction interface (microphone + speech output), while SER inference runs on a host computer using the ONNX model.
Dataset audio and trained checkpoints are not distributed in this repo.
Use `onnx_inference.py` on robot platform with audio input to recognize emotions.

---
## One important repo submission note (based on your screenshots)
Right now your `models/` folder contains **many** `.pt` checkpoints. They are not huge individually (~8–10 MB), but together they can make the repo heavy.

For a clean submission repo, a common approach is:

- keep only **one final checkpoint per track** inside `models/mfcc_v1/` and `models/ssl_v1/`
- keep plots (`curves_*.png`) and `train_summary.json` (these are small and useful)
- ignore the rest using `.gitignore`

If you want, paste your current `.gitignore` (or confirm you don’t have one), and I will write the exact ignore rules for:
- `datasets/*`
- most `models/*.pt` except maybe “final” ones
- caches like `__pycache__` and logs
---

© 2026 – Thesis Project by Zaheen Fatima

---