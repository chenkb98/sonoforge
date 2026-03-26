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


def render(settings: dict, model: dict | None = None) -> NDArray[np.float64]:
    """Render a validated generator configuration into mono audio."""
    values = config(settings)
    name, seconds, rate = values["generator"], values["duration"], values["sample_rate"]
    if name == "noise":
        return synthesis.noise(seconds, rate, values["seed"], values["amplitude"])
    if name == "chirp":
        return synthesis.chirp(
            values["frequency"], values["end_frequency"], seconds, rate, values["amplitude"]
        )
    if name == "diffusion":
        if model is None:
            raise ValueError("diffusion generation requires an explicit model")
        size = len(synthesis.timebase(seconds, rate))
        if size > 32768:
            raise ValueError("baseline diffusion output is limited to 32768 frames")
        if not size:
            return np.empty((0, 1))
        parts, total = [], 0
        while total < size:
            part = diffusion.sample(
                model, diffusion.linear_schedule(values["steps"]), values["seed"] + len(parts)
            )
            parts.append(part)
            total += len(part)
        return audio.concatenate(parts)[:size] * values["amplitude"]
    generators = dict(
        sine=synthesis.sine,
        square=synthesis.square,
        sawtooth=synthesis.sawtooth,
        triangle=synthesis.triangle,
    )
    return generators[name](values["frequency"], seconds, rate, values["amplitude"])


def sweep(base: dict, seeds: list[int], frequencies: list[float]) -> list[dict]:
    """Build a bounded Cartesian parameter sweep with deterministic ordering."""
    original = config(base)
    if not seeds or not frequencies or len(seeds) * len(frequencies) > 1000:
        raise ValueError("sweep needs 1..1000 parameter combinations")
    return [
        config(dict(original, seed=seed, frequency=frequency))
        for seed in seeds
        for frequency in frequencies
    ]


def audio_digest(samples: ArrayLike, rate: int) -> str:
    """Hash canonical PCM16 WAV bytes, including sample rate and channels."""
    import hashlib

    return hashlib.sha256(audio.wav_encode(samples, rate)).hexdigest()
