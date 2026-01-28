import sys
import unittest
import numpy as np
import os
from unittest.mock import MagicMock, patch

from PyQt5.QtWidgets import QApplication
_qt_app = QApplication.instance() or QApplication(sys.argv)

import gui_live_predict as appmodule
print(f"Testing module: {appmodule.__file__}")


# ----------------------------
# Stubs to avoid heavy deps
# ----------------------------
class DummyASR:
    def __init__(self, *args, **kwargs):
        pass
    def transcribe(self, y, sr):
        return "I am feeling great"

class DummySession:
    def run(self, *args, **kwargs):
        # return logits [happy, sad]
        return [np.array([[5.0, 1.0]], dtype=np.float32)]


def _make_safe_app():
    pepper_cfg = {"enabled": False, "use_pepper_mic": False, "ip": "0.0.0.0", "port": 0}

    # Patch away: Whisper/ASR load, ONNX auto-load, Pepper robot mode
    with patch.object(appmodule, "LocalASR", DummyASR), \
         patch.object(appmodule.EmotionApp, "_auto_load_model", lambda self: None), \
         patch.object(appmodule, "PEPPER", pepper_cfg), \
         patch.object(appmodule, "DEBUG_AUDIO", False, create=True):

        gui = appmodule.EmotionApp()

    # Inject minimal runtime fields the worker needs
    gui.session = DummySession()
    gui.input_name = "input"
    gui.classes = ["happy", "sad"]
    gui.model_type = "mfcc"
    gui.temperature = 1.0

    if hasattr(gui, "dialog_phase"):
        gui.dialog_phase = "opener"

    # Replace side-effect methods/signals with mocks
    gui._say = MagicMock()
    gui._add_msg_safe = MagicMock()

    gui.sig_add_emoji = MagicMock()
    gui.sig_add_emoji.emit = MagicMock()
    gui.sig_update_status = MagicMock()
    gui.sig_update_status.emit = MagicMock()
    gui.on_prediction = MagicMock()

    # Stub feature extraction (shape only)
    gui._feat_mfcc = MagicMock(return_value=np.zeros((10, 13), dtype=np.float32))

    # Ensure calib exists
    if not hasattr(gui, "calib") or gui.calib is None:
        gui.calib = {"record_seconds": 5.0}
    else:
        gui.calib.setdefault("record_seconds", 5.0)

    return gui

# ===============================
# FUNCTIONAL TESTS (Thesis §7.5)
# ===============================
class TestThesisFunctional(unittest.TestCase):

    def test_FR6_quiet_audio(self):
        gui = _make_safe_app()

        sr = 16000
        if hasattr(appmodule, "FEATURE_SETTINGS") and isinstance(appmodule.FEATURE_SETTINGS, dict):
            sr = int(appmodule.FEATURE_SETTINGS.get("sample_rate", sr))

        dur = float(gui.calib.get("record_seconds", 5.0))
        n = int(sr * dur)

        with patch.object(appmodule, "sd") as mock_sd, \
             patch.object(appmodule, "remove_fan_noise", side_effect=lambda y, sr_: y):

            mock_sd.rec.return_value = np.zeros((n, 1), dtype="float32")
            mock_sd.wait.return_value = None

            gui._record_and_predict_worker()

            gui._say.assert_called_with("I couldn't hear you.")
            gui.sig_add_emoji.emit.assert_not_called()

        gui.close()

    def test_FR2_happy_flow(self):
        gui = _make_safe_app()

        sr = 16000
        if hasattr(appmodule, "FEATURE_SETTINGS") and isinstance(appmodule.FEATURE_SETTINGS, dict):
            sr = int(appmodule.FEATURE_SETTINGS.get("sample_rate", sr))

        dur = float(gui.calib.get("record_seconds", 5.0))
        n = int(sr * dur)

        gui.asr.transcribe = MagicMock(return_value="I am feeling great")

        with patch.object(appmodule, "USE_SENTIMENT_FUSION", False, create=True), \
             patch.object(appmodule, "sd") as mock_sd, \
             patch.object(appmodule, "remove_fan_noise", side_effect=lambda y, sr_: y):

            mock_sd.rec.return_value = np.ones((n, 1), dtype="float32") * 0.5
            mock_sd.wait.return_value = None

            gui._record_and_predict_worker()

            gui.sig_add_emoji.emit.assert_called_with("happy")
            gui._say.assert_called()

        gui.close()
    
    def test_FR_multimodal_fusion(self):
        """
        Test Multimodal Fusion:
        Scenario: User sounds SAD (Audio) but says "I am so happy" (Text).
        Expected: The Text Sentiment should override the Audio to predict 'happy'.
        """
        gui = _make_safe_app()
        
        # 1. Simulate "SAD" Audio (Model predicts Sad > Happy)
        # We mock the session to return [Happy=1.0, Sad=5.0]
        gui.session.run = MagicMock(return_value=[np.array([[1.0, 5.0]], dtype=np.float32)])
        
        # 2. Simulate "HAPPY" Text
        gui.asr.transcribe = MagicMock(return_value="I am so happy today")
        
        # 3. Enable Fusion Logic
        # We patch USE_SENTIMENT_FUSION to True
        with patch.object(appmodule, "USE_SENTIMENT_FUSION", True), \
             patch.object(appmodule, "sd") as mock_sd, \
             patch.object(appmodule, "remove_fan_noise", side_effect=lambda y, sr: y):
            
            # Record dummy audio
            mock_sd.rec.return_value = np.ones((16000*3, 1), dtype="float32") * 0.5
            mock_sd.wait.return_value = None
            
            # Run Worker
            gui._record_and_predict_worker()
            
            # 4. Verification
            # Even though audio model said SAD, the TextBlob sentiment (Positive) 
            # should have flipped the final result to HAPPY.
            gui.sig_add_emoji.emit.assert_called_with("happy")
            
        gui.close()
        
    def test_FR2_sad_detection(self):
        """
        Test FR2: Valid Audio -> Sad Prediction.
        Scenario: The model detects 'Sad' (Sad prob > Happy prob).
        Expected: Robot emits 'sad' signal and speaks supportive response.
        """
        gui = _make_safe_app()
        n_samples = int(16000 * 3.0)

        # 1. Mock ASR (User says something sad)
        gui.asr.transcribe = MagicMock(return_value="I am feeling down")

        # 2. Mock AI Model to return SAD Logic
        # Logits: [Happy=1.0, Sad=5.0] -> Sad is dominant
        gui.session.run = MagicMock(return_value=[np.array([[1.0, 5.0]], dtype=np.float32)])

        # 3. Disable Sentiment Fusion (Focus on Audio only)
        with patch.object(appmodule, "USE_SENTIMENT_FUSION", False):
            with patch.object(appmodule, "sd") as mock_sd:
                mock_sd.rec.return_value = np.ones((n_samples, 1), dtype="float32") * 0.5
                mock_sd.wait.return_value = None

                with patch.object(appmodule, "remove_fan_noise", side_effect=lambda y, sr: y):
                    gui._record_and_predict_worker()

                # 4. Verify the Robot chose "Sad"
                gui.sig_add_emoji.emit.assert_called_with("sad")
                gui._say.assert_called()

        gui.close()

    def test_real_mfcc_onnx_inference_runs(self):
        # Use your real model from repo
        model_path = os.path.join("models", "mfcc_v1", "model_mfcc.onnx")
        wav_path = "debug_pepper.wav"   # you already have this in the repo

        if not os.path.exists(model_path):
            self.skipTest(f"Missing model: {model_path}")
        if not os.path.exists(wav_path):
            self.skipTest(f"Missing wav: {wav_path}")

        try:
            import onnxruntime as ort
        except Exception:
            self.skipTest("onnxruntime is not installed in this environment")

        from extract_features import extract_mfcc

        # 1) real MFCC features
        feats = extract_mfcc(audio_path=wav_path)          # (T, D)
        x = feats[np.newaxis, :, :].astype(np.float32)     # (1, T, D)

        # 2) real ONNX inference
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: x})[0][0]     # (2,)

        # 3) convert to probabilities + label
        probs = appmodule.softmax(np.array(logits, dtype=np.float32))
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

        label = ["happy", "sad"][int(np.argmax(probs))]
        print(f"[REAL ONNX] label={label}, probs={probs}")
        self.assertIn(label, ["happy", "sad"])

   

if __name__ == "__main__":
    print("Running thesis unittest (Chapter 7)...")
    unittest.main(verbosity=2)