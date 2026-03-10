from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio, diffusion, synthesis


def config(values: dict | None = None) -> dict:
    """Validate a strict, serializable generation configuration."""
    result = dict(
        generator="sine",
        sample_rate=16000,
        duration=0.1,
        frequency=440.0,
        end_frequency=880.0,
        amplitude=0.5,
        seed=0,
        steps=20,
    )
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(result):
            raise ValueError("unknown configuration fields")
        result.update(values)
    if result["generator"] not in [
        "sine",
        "square",
        "sawtooth",
        "triangle",
        "noise",
        "chirp",
        "diffusion",
    ]:
        raise ValueError("unknown generator")
    audio.sample_rate(result["sample_rate"])
    for key in ["duration", "frequency", "end_frequency", "amplitude"]:
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], (int, float))
            or not math.isfinite(result[key])
        ):
            raise ValueError("configuration numbers must be finite real scalars")
    synthesis.timebase(result["duration"], result["sample_rate"])
    synthesis.sine(result["frequency"], 0, result["sample_rate"], result["amplitude"])
    if result["generator"] == "chirp":
        synthesis.sine(result["end_frequency"], 0, result["sample_rate"])
    if (
        isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or not 0 <= result["seed"] < 2**63
    ):
        raise ValueError("invalid seed")
    diffusion.linear_schedule(result["steps"])
    return result


def parse_prompt(prompt: str) -> dict:
    """Parse a small symbolic prompt grammar, not natural-language semantics."""
    if not isinstance(prompt, str) or len(prompt) > 200:
        raise ValueError("prompt must be a short string")
    tokens = prompt.strip().lower().split()
    if tokens == ["noise"]:
        return {"generator": "noise"}
    if len(tokens) == 2 and tokens[0] in ["sine", "square", "sawtooth", "triangle"]:
        values = {"generator": tokens[0], "frequency": float(tokens[1])}
    elif len(tokens) == 3 and tokens[0] == "chirp":
        values = {
            "generator": "chirp",
            "frequency": float(tokens[1]),
            "end_frequency": float(tokens[2]),
        }
    else:
        raise ValueError("expected noise, GENERATOR HZ, or chirp START_HZ END_HZ")
    config(values)
    return values
