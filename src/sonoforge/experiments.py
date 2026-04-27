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


def manifest(settings: dict, samples: ArrayLike) -> dict:
    """Describe a generated mono artifact with exact configuration and diagnostics."""
    values, array = config(settings), audio.as_audio(samples)
    expected = len(synthesis.timebase(values["duration"], values["sample_rate"]))
    if array.shape != (expected, 1):
        raise ValueError("artifact shape disagrees with generation configuration")
    return {
        "schema_version": 1,
        "config": values,
        "frames": len(array),
        "channels": 1,
        "sha256": audio_digest(array, values["sample_rate"]),
        "peak": audio.peak(array),
        "rms": audio.rms(array),
    }


def _manifest(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "config",
        "frames",
        "channels",
        "sha256",
        "peak",
        "rms",
    }:
        raise ValueError("invalid manifest fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["channels"]) is not int
        or value["channels"] != 1
    ):
        raise ValueError("unsupported manifest schema")
    settings = config(value["config"])
    if set(value["config"]) != set(settings):
        raise ValueError("manifest configuration must be complete")
    frames = len(synthesis.timebase(settings["duration"], settings["sample_rate"]))
    if type(value["frames"]) is not int or value["frames"] != frames:
        raise ValueError("invalid frame count")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid SHA-256")
    for key in ["peak", "rms"]:
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], (int, float))
            or not math.isfinite(value[key])
            or value[key] < 0
        ):
            raise ValueError("invalid audio diagnostic")
    if value["rms"] > value["peak"] + 1e-12:
        raise ValueError("RMS cannot exceed peak")
    return dict(value, config=settings)


def manifest_encode(value: dict) -> str:
    """Serialize the exact manifest schema to stable, finite JSON."""
    import json

    return (
        json.dumps(
            _manifest(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def manifest_decode(text: str) -> dict:
    """Parse a manifest, rejecting duplicate keys and nonstandard constants."""
    import json

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("non-finite JSON constant")

    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError("manifest must be a bounded string")
    return _manifest(json.loads(text, object_pairs_hook=pairs, parse_constant=constant))


def generate_run(configurations: list[dict], model: dict | None = None) -> list[dict]:
    """Render a bounded list of configurations and retain paired manifests."""
    if not isinstance(configurations, list) or not 1 <= len(configurations) <= 1000:
        raise ValueError("run needs 1..1000 configurations")
    if any(not isinstance(value, dict) for value in configurations):
        raise ValueError("each configuration must be an object")
    settings = [config(value) for value in configurations]
    result = []
    for value in settings:
        samples = render(value, model)
        result.append({"manifest": manifest(value, samples), "audio": samples})
    return result


def write_run(directory: str, results: list[dict]) -> list[str]:
    """Write a new result directory; reject mismatched artifacts and existing paths."""
    from pathlib import Path
    import json
    import shutil

    if not isinstance(results, list) or not 1 <= len(results) <= 1000:
        raise ValueError("results must be a bounded nonempty list")
    artifacts, records = [], []
    for index, result in enumerate(results):
        if not isinstance(result, dict) or set(result) != {"manifest", "audio"}:
            raise ValueError("invalid result schema")
        metadata = _manifest(result["manifest"])
        if manifest(metadata["config"], result["audio"]) != metadata:
            raise ValueError("artifact content disagrees with manifest")
        filename = f"{index:04d}-{metadata['sha256'][:12]}.wav"
        artifacts.append(
            (filename, audio.wav_encode(result["audio"], metadata["config"]["sample_rate"]))
        )
        records.append({"file": filename, "manifest": metadata})
    target = Path(directory)
    target.mkdir(exist_ok=False)
    try:
        for filename, payload in artifacts:
            (target / filename).write_bytes(payload)
        (target / "manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
    except Exception:
        shutil.rmtree(target)
        raise
    return [str(target / name) for name, _ in artifacts]
