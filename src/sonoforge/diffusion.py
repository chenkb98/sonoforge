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


def validate_schedule(betas: ArrayLike) -> NDArray[np.float64]:
    """Copy a valid nonempty beta vector and verify usable cumulative alphas."""
    raw = np.asarray(betas)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or not 1 <= len(raw) <= 10000:
        raise ValueError("betas must be a bounded real numeric vector")
    values = np.array(raw, dtype=float, copy=True)
    if (
        not np.isfinite(values).all()
        or np.any((values <= 0) | (values >= 1))
        or np.prod(1 - values) < 1e-15
    ):
        raise ValueError("schedule has invalid variances or unusably small cumulative alpha")
    return values


def forward_noise(clean: ArrayLike, noise: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Mix explicit clean samples and epsilon at a cumulative signal fraction."""
    a, b = np.asarray(clean, dtype=float), np.asarray(noise, dtype=float)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("noise mixing requires finite equal shapes and alpha in [0, 1]")
    return np.sqrt(alpha) * a + np.sqrt(1 - alpha) * b


def predict_clean(noisy: ArrayLike, epsilon: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Invert the forward process for a given epsilon prediction."""
    a, b = np.asarray(noisy, dtype=float), np.asarray(epsilon, dtype=float)
    forward_noise(a, b, alpha)
    if alpha < 1e-15:
        raise ValueError("alpha is too small to invert")
    return (a - np.sqrt(1 - alpha) * b) / np.sqrt(alpha)


def ddim_step(
    noisy: ArrayLike, epsilon: ArrayLike, alpha: float, previous_alpha: float
) -> NDArray[np.float64]:
    """Perform a deterministic DDIM step with eta=0 and explicit alphas."""
    clean = predict_clean(noisy, epsilon, alpha)
    if not np.isfinite(previous_alpha) or not alpha <= previous_alpha <= 1:
        raise ValueError("previous alpha must lie between current alpha and one")
    return forward_noise(clean, epsilon, previous_alpha)


def _model(model: dict) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(model, dict) or set(model) != {"weights", "bias"}:
        raise ValueError("model fields must be weights and bias")
    weights, bias = np.asarray(model["weights"]), np.asarray(model["bias"])
    if (
        weights.dtype.kind not in "iuf"
        or bias.dtype.kind not in "iuf"
        or weights.ndim != 2
        or bias.ndim != 1
        or not 1 <= len(bias) <= 256
        or weights.shape != (len(bias) + 1, len(bias))
    ):
        raise ValueError("invalid affine denoiser shapes")
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("model parameters must be finite")
    return weights.astype(float), bias.astype(float)


def initialize_model(width: int = 16, seed: int = 0) -> dict:
    """Initialize a tiny time-conditioned affine epsilon predictor."""
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 256:
        raise ValueError("width must be in 1..256")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("invalid seed")
    rng = np.random.default_rng(seed)
    return {"weights": rng.normal(0, 0.001, (width + 1, width)), "bias": np.zeros(width)}


def _batch(noisy: ArrayLike, times: ArrayLike, width: int) -> NDArray[np.float64]:
    values = np.asarray(noisy, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    time = np.asarray(times, dtype=float)
    if time.ndim == 0:
        time = np.full(len(values), float(time))
    if (
        values.ndim != 2
        or values.shape[1] != width
        or time.shape != (len(values),)
        or not np.isfinite(values).all()
        or not np.isfinite(time).all()
        or np.any((time < 0) | (time > 1))
    ):
        raise ValueError("batch must match the model width with times in [0, 1]")
    return np.column_stack([values, time])


def predict_noise(model: dict, noisy: ArrayLike, times: ArrayLike) -> NDArray[np.float64]:
    """Predict epsilon for a batch of short audio windows and normalized times."""
    weights, bias = _model(model)
    inputs = _batch(noisy, times, len(bias))
    result = inputs @ weights + bias
    if not np.isfinite(result).all():
        raise ValueError("prediction overflowed")
    return result


def training_loss(
    model: dict, noisy: ArrayLike, times: ArrayLike, target_noise: ArrayLike
) -> float:
    """Mean squared epsilon prediction error on a nonempty batch."""
    prediction = predict_noise(model, noisy, times)
    target = np.asarray(target_noise, dtype=float)
    if target.shape != prediction.shape or not target.size or not np.isfinite(target).all():
        raise ValueError("targets must be finite, nonempty and match predictions")
    error = prediction - target
    result = float(np.mean(error**2))
    if not np.isfinite(result):
        raise ValueError("loss overflowed")
    return result
