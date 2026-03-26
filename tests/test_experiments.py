import numpy as np
import pytest
from sonoforge import experiments as m


def test_config_example():
    value = m.config()
    assert value["generator"] == "sine" and value["sample_rate"] == 16000


def test_config_boundaries():
    for values in [
        {"unknown": 1},
        {"generator": "magic"},
        {"duration": np.nan},
        {"sample_rate": True},
        {"amplitude": 2},
        {"seed": -1},
        {"steps": 0},
        {"frequency": True},
    ]:
        with pytest.raises(ValueError):
            m.config(values)


def test_config_invariants():
    expected = {
        "generator",
        "sample_rate",
        "duration",
        "frequency",
        "end_frequency",
        "amplitude",
        "seed",
        "steps",
    }
    assert set(m.config()) == expected
    for key in expected:
        candidate = m.config()
        candidate[key + "_typo"] = candidate[key]
        with pytest.raises(ValueError):
            m.config(candidate)
    value = m.config({"seed": 7})
    assert m.config(value) == value


def test_parse_prompt_example():
    assert m.parse_prompt("sine 440") == {"generator": "sine", "frequency": 440.0}


def test_parse_prompt_boundaries():
    for prompt in ["", "a dog barking", "sine", "sine nan", "noise 2", "chirp 10", "sine -1"]:
        with pytest.raises(ValueError):
            m.parse_prompt(prompt)


def test_parse_prompt_invariants():
    assert m.parse_prompt("  SINE   440 ") == m.parse_prompt("sine 440")
    assert m.parse_prompt("noise") == {"generator": "noise"}
    assert m.parse_prompt("chirp 220 880")["end_frequency"] == 880
    for name in ["sine", "square", "sawtooth", "triangle"]:
        assert m.parse_prompt(name + " 440")["generator"] == name


def test_render_example():
    np.testing.assert_allclose(
        m.render({"sample_rate": 8, "frequency": 1, "duration": 1, "amplitude": 1}).ravel(),
        np.sin(2 * np.pi * np.arange(8) / 8),
    )


def test_render_boundaries():
    with pytest.raises(ValueError):
        m.render({"generator": "diffusion"})
    with pytest.raises(ValueError):
        m.render({"generator": "unknown"})
    assert m.render({"duration": 0}).shape == (0, 1)


def test_render_invariants():
    from sonoforge import diffusion

    for name in ["sine", "square", "sawtooth", "triangle", "noise", "chirp", "diffusion"]:
        settings = {"generator": name, "duration": 0.001, "seed": 7}
        model = diffusion.initialize_model(8)
        x = m.render(settings, model)
        assert x.shape == (16, 1) and np.max(np.abs(x)) <= 0.5
        np.testing.assert_array_equal(x, m.render(settings, model))


def test_sweep_example():
    values = m.sweep({}, [1, 2], [220, 440])
    assert [(x["seed"], x["frequency"]) for x in values] == [(1, 220), (1, 440), (2, 220), (2, 440)]


def test_sweep_boundaries():
    for seeds, frequencies in [
        ([], [220]),
        ([1], []),
        ([-1], [220]),
        ([1], [np.nan]),
        (list(range(1001)), [220]),
    ]:
        with pytest.raises(ValueError):
            m.sweep({}, seeds, frequencies)


def test_sweep_invariants():
    base = {"amplitude": 0.2}
    values = m.sweep(base, [1, 2], [100])
    assert base == {"amplitude": 0.2}
    values[0]["amplitude"] = 0.9
    assert values[1]["amplitude"] == 0.2
    assert m.sweep(base, [1, 2], [100]) == m.sweep(base, [1, 2], [100])


def test_audio_digest_example():
    assert len(m.audio_digest([0, 1], 16000)) == 64


def test_audio_digest_boundaries():
    with pytest.raises(ValueError):
        m.audio_digest([np.nan], 16000)
    with pytest.raises(ValueError):
        m.audio_digest([0], 0)
    assert len(m.audio_digest([], 16000)) == 64
