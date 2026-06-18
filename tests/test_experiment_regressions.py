import numpy as np
import pytest

from sonoforge import audio, diffusion, experiments


def test_prompt_uses_the_selected_sample_rate_in_both_directions():
    assert experiments.parse_prompt("sine 10000", 48000)["frequency"] == 10000
    for prompt, rate in [("sine 10000", 16000), ("sine 6000", 8000), ("noise", 0)]:
        with pytest.raises(ValueError):
            experiments.parse_prompt(prompt, rate)


def test_manifest_diagnostics_describe_exact_serialized_pcm():
    settings = dict(sample_rate=4, frequency=1, duration=1)
    samples = [2.0, -2.0, 0.123456789, -0.123456789]
    value = experiments.manifest(settings, samples)
    decoded, _ = audio.wav_decode(audio.wav_encode(samples, 4))
    assert value["peak"] == audio.peak(decoded)
    assert value["rms"] == audio.rms(decoded)
    assert value["sha256"] == experiments.audio_digest(decoded, 4)
    assert experiments.manifest(settings, decoded) == value


def test_diffusion_windows_wrap_a_valid_maximum_seed():
    settings = dict(
        generator="diffusion", sample_rate=8, frequency=1, duration=1, seed=2**63 - 1, steps=2
    )
    model = diffusion.initialize_model(4)
    result = experiments.render(settings, model)
    np.testing.assert_array_equal(result, experiments.render(settings, model))
    assert result.shape == (8, 1) and np.isfinite(result).all()


@pytest.mark.parametrize("noisy", [0.0, np.zeros((1, 1, 1))])
def test_denoiser_rejects_invalid_ranks_consistently(noisy):
    with pytest.raises(ValueError):
        diffusion.predict_noise(diffusion.initialize_model(1), noisy, 0.5)
