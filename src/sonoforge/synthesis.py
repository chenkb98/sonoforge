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


def noise(
    seconds: float, rate: int = 16000, seed: int = 0, amplitude: float = 0.5
) -> NDArray[np.float64]:
    """Generate seeded uniform white noise without changing NumPy global RNG."""
    clock = timebase(seconds, rate)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("invalid amplitude")
    return as_audio(np.random.default_rng(seed).uniform(-amplitude, amplitude, len(clock)))


def chirp(
    start_hz: float, end_hz: float, seconds: float, rate: int = 16000, amplitude: float = 0.5
) -> NDArray[np.float64]:
    """Generate a linear frequency sweep using the integrated phase."""
    sine(start_hz, 0, rate, amplitude)
    sine(end_hz, 0, rate, amplitude)
    clock = timebase(seconds, rate)
    slope = (end_hz - start_hz) / seconds if seconds else 0.0
    return as_audio(amplitude * np.sin(2 * np.pi * (start_hz * clock + 0.5 * slope * clock**2)))


def harmonics(
    fundamental: float, weights: ArrayLike, seconds: float, rate: int = 16000
) -> NDArray[np.float64]:
    """Add harmonics below Nyquist, normalized by total absolute weight."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("weights must be a nonempty finite vector")
    sine(fundamental, 0, rate)
    if fundamental <= 0 or fundamental * len(values) >= rate / 2:
        raise ValueError("all harmonics must be below Nyquist")
    result = np.zeros((len(timebase(seconds, rate)), 1))
    for index, weight in enumerate(values, 1):
        result += weight * sine(fundamental * index, seconds, rate, 1)
    total = float(np.abs(values).sum())
    return result / total if total else result
