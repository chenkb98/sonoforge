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


def square(
    frequency: float, seconds: float, rate: int = 16000, amplitude: float = 0.5
) -> NDArray[np.float64]:
    """Generate a simple square oscillator; higher harmonics are not bandlimited."""
    sine(frequency, 0, rate, amplitude)
    clock = timebase(seconds, rate)
    cycles = frequency * clock
    return as_audio(amplitude * (np.where(cycles - np.floor(cycles) <= 0.5, 1.0, -1.0)))


def sawtooth(
    frequency: float, seconds: float, rate: int = 16000, amplitude: float = 0.5
) -> NDArray[np.float64]:
    """Generate a simple sawtooth oscillator; higher harmonics are not bandlimited."""
    sine(frequency, 0, rate, amplitude)
    clock = timebase(seconds, rate)
    cycles = frequency * clock
    return as_audio(amplitude * (2 * (cycles - np.floor(cycles)) - 1))


def triangle(
    frequency: float, seconds: float, rate: int = 16000, amplitude: float = 0.5
) -> NDArray[np.float64]:
    """Generate a simple triangle oscillator; higher harmonics are not bandlimited."""
    sine(frequency, 0, rate, amplitude)
    clock = timebase(seconds, rate)
    cycles = frequency * clock
    return as_audio(amplitude * (1 - 4 * np.abs(cycles - np.floor(cycles) - 0.5)))
