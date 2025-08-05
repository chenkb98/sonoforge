from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from .audio import as_audio, sample_rate


def timebase(seconds: float, rate: int) -> NDArray[np.float64]:
    """Create a half-open sample clock with nearest-integer duration rounding."""
    rate = sample_rate(rate)
    if not np.isfinite(seconds) or seconds < 0 or seconds * rate > 10000000:
        raise ValueError("duration must be finite, nonnegative and bounded")
    return np.arange(int(math.floor(seconds * rate + 0.5)), dtype=np.float64) / rate


def sine(
    frequency: float, seconds: float, rate: int = 16000, amplitude: float = 0.5, phase: float = 0.0
) -> NDArray[np.float64]:
    """Generate a finite sine below Nyquist."""
    clock = timebase(seconds, rate)
    if (
        not np.isfinite([frequency, amplitude, phase]).all()
        or not 0 <= frequency < rate / 2
        or not 0 <= amplitude <= 1
    ):
        raise ValueError("invalid oscillator frequency, amplitude or phase")
    return as_audio(amplitude * np.sin(2 * np.pi * frequency * clock + phase))
