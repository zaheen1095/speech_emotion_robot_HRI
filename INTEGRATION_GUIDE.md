# INTEGRATION_GUIDE.md
# SER Component Integration Guide (Happy vs Sad)

This document explains how to **verify** and **integrate** the Speech Emotion Recognition (SER) component from this repository into another software system (for example, a mental-health chatbot platform or a robot interaction stack).

This guide is for developers who want to **use the trained component for inference**, not retrain it.

---

## 1. What this component provides

This SER component predicts a **binary speech affect label**:

- `happy`
- `sad`

In addition, the runtime may return:

- `uncertain`

`uncertain` is not a trained third class. It is produced at runtime when the input is weak (silence/too quiet) or when confidence is too low. This avoids forcing a label on unreliable input.

**Important scope note:** this component outputs a speech-based affect label only. It is not a clinical tool and it does not infer mental health state.

---

## 2. Integration concept (loose coupling)

The host system (chatbot or robot software) should treat SER as a **separate module** with a simple contract.

**Input:** an audio clip  
**Output:** `{label, confidence, p_happy, p_sad}`

The host system decides how to use the label (for example, choosing a supportive reply style when `sad` is detected). The SER component does not call chatbot code and does not generate dialogue.

This keeps integration loosely coupled because the host system does **not** need to know:
- training scripts and hyperparameters
- dataset structure
- augmentation logic
- model architecture details

---

## 3. Files needed for integration (minimum set)

### Required
- An exported ONNX model (one of these must exist):
  - `models/ssl_v1/model_ssl.onnx`
  - `models/mfcc_v1/model_mfcc.onnx`
- `config.py` (feature settings / runtime thresholds)
- `ser_component.py` (wrapper API for integration)

### Optional
- Calibration settings (if present):
  - `models/*/calibration.json` (threshold / temperature / confidence gate)

> Datasets are not needed for inference and are not included in the repository.

---

## 4. Quick check (how to verify the component works)

### Step 1 — Confirm an ONNX model exists
Check that one of these files is present:

- `models/ssl_v1/model_ssl.onnx`
- `models/mfcc_v1/model_mfcc.onnx`

If neither exists, export the model first (after training):

```bash
python export_to_onnx.py
```

### Step 2 — Run the wrapper on one WAV file
Use a short WAV file as input (recommended: 16 kHz mono, 2–5 seconds).
If ser_component.py includes a small CLI test block, run:
```bash
python ser_component.py --wav path/to/sample.wav
```
Expected output (example):
- a printed object or JSON with:

    - label in {happy, sad, uncertain}

    - probability scores and confidence

If the output is uncertain, try a louder clip or a longer clip to confirm it is not just a silence/quiet issue.
---

## 5. Integration options (choose one)
### Option A (recommended): Python integration (direct import)

This is the cleanest option when the host system is Python-based.

Host system code (example):
```python 
from ser_component import SERComponent

ser = SERComponent(models_root="models", prefer_ssl=True)

out = ser.predict_wav("user_turn.wav")

# out.label: "happy" | "sad" | "uncertain"
# out.confidence: float
# out.prob_happy / out.prob_sad: float

if out.label == "sad":
    # choose supportive response style
    pass
elif out.label == "happy":
    # normal/positive response style
    pass
else:
    # uncertain: ask user to repeat / speak louder
    pass
```
This keeps the SER module loosely coupled: the chatbot only calls predict_*() and uses the returned label.
---
### Option B: CLI / subprocess integration (any programming language)

This is useful when the host system is not Python (Node.js, Java, Unity, etc.).

The host system should:

1. record and save audio as a WAV file

2. call the SER command

3. parse the printed JSON output

Recommended command (after adding a small CLI entry point such as ser_component.py):
```bash
    python ser_component.py --wav path/to/user_turn.wav
```
Expected JSON output (example):
```json
    {"label":"sad","confidence":0.81,"prob_sad":0.81,"prob_happy":0.19}
```
---
### Option C: Local HTTP service

This option runs SER as a separate local service. The host system sends audio to an endpoint and receives JSON back.

Typical endpoints:

- GET /health

- POST /predict (body: WAV bytes)

This service file is not included by default, but it can be added later because inference is already separated from training in this repo.
---

## 6. Input details (audio)
#### Recommended audio format
- mono

- 16 kHz

- WAV PCM is easiest for integration

- clip length: typically 2–5 seconds works well

#### Silence / weak audio

If the clip is mostly silence or too quiet, the component may return uncertain. This is expected behaviour and is used to avoid forcing labels on weak input.
---

## 7. Output details
For each input turn, the SER component should return:

- label: "happy" | "sad" | "uncertain"

- p_happy: probability score for happy

- p_sad: probability score for sad

- confidence (usually max(p_happy, p_sad))

- optional: reason for uncertain (e.g., "too_quiet", "low_confidence")
---


## 8. Common integration mistakes
1. Wrong sampling rate

- Fix: resample to 16 kHz mono (or use wrapper that resamples internally)

2. Stereo audio

- Fix: convert to mono before inference

3. Missing packages

Fix: install requirements using:
```bash
pip install -r requirements.txt
```
4. Mixing training scripts with runtime

- Fix: integration should only depend on ONNX + wrapper (ser_component.py), not on training scripts.
---

## 9. Notes for multi-emotion mental-health systems

If the host project supports many emotions, this SER component can still be used as an extra speech-based affect cue:

- a simple positive vs negative signal

- a fallback when multi-class output is uncertain

- an auxiliary input for response style selection

However, it should be treated as binary affect only, because it was trained and evaluated only for happy vs sad.
---


© 2026 – Thesis Project by Zaheen Fatima

