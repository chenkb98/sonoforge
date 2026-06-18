from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import audio


def stable_audio(
    prompt: str, pipeline: Any, generator: Any, seconds: float = 5.0, steps: int = 50
) -> dict:
    """Invoke an explicitly supplied Diffusers StableAudioPipeline and generator."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000:
        raise ValueError("prompt must be a nonempty bounded string")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or not 0 < seconds <= 600
    ):
        raise ValueError("duration must be finite and in (0, 600]")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError("inference steps must be in 1..1000")
    if not callable(pipeline) or generator is None:
        raise ValueError("a loaded pipeline and explicit random generator are required")
    vae = getattr(pipeline, "vae", None)
    rate = getattr(vae, "sampling_rate", None)
    if rate is None:
        rate = getattr(getattr(vae, "config", None), "sampling_rate", None)
    if rate is None:
        raise ValueError("pipeline VAE must declare its sampling rate")
    rate = audio.sample_rate(rate)
    result = pipeline(
        prompt=prompt,
        generator=generator,
        audio_end_in_s=seconds,
        num_inference_steps=steps,
        num_waveforms_per_prompt=1,
    )
    values = result.audios
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] == 0:
        raise ValueError("pipeline must return one (batch, channels, samples) waveform")
    return {"audio": audio.as_audio(array[0].T), "sample_rate": rate}
