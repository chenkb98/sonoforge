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
