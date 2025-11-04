from pepper_client import PepperClient
import soundfile as sf, io, numpy as np
pc = PepperClient("192.168.0.3", 7878); pc.connect()
raw = pc.record(3)                                  # 3 seconds
y, sr = sf.read(io.BytesIO(raw), dtype="float32")
if y.ndim > 1: y = y.mean(axis=1)
print("frames:", len(y), "sr:", sr, "max_amp:", float(np.abs(y).max()))
sf.write("pepper_3s.wav", y, sr)                    # listen to confirm
pc.close()
