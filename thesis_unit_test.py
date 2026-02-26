import sys
import unittest
import numpy as numpy
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
    def transcribe(self, y, sample_rate_hz):
        return "I am feeling great"

class DummySession:
    def run(self, *args, **kwargs):
        # return logits [happy, sad]
        return [numpy.array([[5.0, 1.0]], dtype=numpy.float32)]


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
    gui._feat_mfcc = MagicMock(return_value=numpy.zeros((10, 13), dtype=numpy.float32))

    # Ensure calib exists
    if not hasattr(gui, "calib") or gui.calib is None:
        gui.calib = {"record_seconds": 5.0}
    else:
        gui.calib.setdefault("record_seconds", 5.0)

    return gui


# ============================
# UNIT TESTS
# ============================
class TestThesisUnitChecks(unittest.TestCase):

    def test_softmax_logic(self):
        iteration_logs = numpy.array([2.0, 1.0, 0.1])
        probability = appmodule.softmax(iteration_logs)
        self.assertAlmostEqual(float(numpy.sum(probability)), 1.0, places=6)
        self.assertGreater(probability[0], probability[1])

    def test_clean_transcript(self):
        gui = _make_safe_app()

        self.assertIsNone(gui._clean_transcript("Thank you for watching"))
        self.assertIsNone(gui._clean_transcript("Subtitle by Amara"))
        self.assertIsNone(gui._clean_transcript("a"))

        # your code returns original stripped text
        self.assertEqual(gui._clean_transcript("I am feeling happy"), "I am feeling happy")
        self.assertEqual(gui._clean_transcript("Sad"), "Sad")

        gui.close()

    def test_remove_fan_noise_reduces_low_freq(self):
        sample_rate_hz = 16000
        time_s = numpy.linspace(0, 1, sample_rate_hz, endpoint=False)
        noise = 0.8 * numpy.sin(2*numpy.pi*50*time_s)
        voice = 0.2 * numpy.sin(2*numpy.pi*500*time_s)
        y = noise + voice

        y_clean = appmodule.remove_fan_noise(y, sample_rate_hz)

        Y = numpy.abs(numpy.fft.rfft(y))
        Yc = numpy.abs(numpy.fft.rfft(y_clean))
        freq_bins_hz = numpy.fft.rfftfreq(len(y), 1/sample_rate_hz)

        def band_mag(arr, f0, bw=3):
            idx = int(numpy.argmin(numpy.abs(freq_bins_hz - f0)))
            lo = max(0, idx-bw)
            hi = min(len(arr), idx+bw+1)
            return float(numpy.mean(arr[lo:hi]))

        self.assertLess(band_mag(Yc, 50), band_mag(Y, 50))
        self.assertGreater(band_mag(Yc, 500), 0.01)


if __name__ == "__main__":
    print("Running thesis unittest (Chapter 7)...")
    unittest.main(verbosity=2)
