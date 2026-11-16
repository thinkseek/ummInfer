# Test-only 24 kHz -> 16 kHz resampler for the ASR round-trip check.
# Not part of the runtime path; the TTS engine itself emits 24 kHz only.
# Usage: python resample16.py in24k.wav out16k.wav
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sr, x = wavfile.read(sys.argv[1])
x = x.astype(np.float32) / 32768.0
g = np.gcd(sr, 16000)
y = resample_poly(x, 16000 // g, sr // g)
y16 = np.clip(y, -1.0, 1.0)
wavfile.write(sys.argv[2], 16000, (y16 * 32767).astype(np.int16))
print("wrote", sys.argv[2], len(y16), "samples")
