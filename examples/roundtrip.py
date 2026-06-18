import numpy as np

from sonoforge.audio import frames, overlap_add, wav_decode, wav_encode
from sonoforge.synthesis import sine

signal = sine(440, 0.05)
rebuilt = overlap_add(frames(signal, 128, 64), 64, len(signal))
np.testing.assert_allclose(rebuilt, signal)
decoded, rate = wav_decode(wav_encode(signal, 16000))
assert np.max(np.abs(decoded - signal)) <= 1 / 32768
print({"frames": len(decoded), "sample_rate": rate, "roundtrip": "passed"})
