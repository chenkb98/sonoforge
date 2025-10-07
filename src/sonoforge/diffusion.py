from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def linear_schedule(steps: int, start: float = 0.0001, end: float = 0.02) -> NDArray[np.float64]:
    """Create a monotonically increasing vector of diffusion noise variances."""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10000:
        raise ValueError("steps must be in 1..10000")
    if not np.isfinite([start, end]).all() or not 0 < start <= end < 1:
        raise ValueError("noise variances must satisfy 0 < start <= end < 1")
    return np.linspace(start, end, steps)


def cosine_schedule(steps: int, offset: float = 0.008) -> NDArray[np.float64]:
    """Discretize a cosine cumulative-alpha schedule, capping beta at 0.999."""
    linear_schedule(steps)
    if not np.isfinite(offset) or not 0 <= offset <= 1:
        raise ValueError("offset must be in [0, 1]")
    time = np.arange(steps + 1) / steps
    alpha = np.cos((time + offset) / (1 + offset) * np.pi / 2) ** 2
    alpha /= alpha[0]
    return np.clip(1 - alpha[1:] / alpha[:-1], 1e-8, 0.999)
