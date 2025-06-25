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
