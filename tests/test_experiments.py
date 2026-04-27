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


def test_audio_digest_invariants():
    import hashlib
    from sonoforge import audio

    x = [0, 0.5, -1]
    assert m.audio_digest(x, 16000) == hashlib.sha256(audio.wav_encode(x, 16000)).hexdigest()
    assert m.audio_digest(x, 16000) != m.audio_digest(x, 24000)
    assert m.audio_digest(x, 16000) != m.audio_digest(np.column_stack([x, x]), 16000)


def test_manifest_example():
    result = m.manifest({"duration": 0}, [])
    assert result["frames"] == 0 and result["peak"] == result["rms"] == 0


def test_manifest_boundaries():
    with pytest.raises(ValueError):
        m.manifest({}, [1])
    with pytest.raises(ValueError):
        m.manifest({"duration": 0}, np.empty((0, 2)))
    with pytest.raises(ValueError):
        m.manifest({"duration": 0.0000625}, [np.nan])


def test_manifest_invariants():
    settings = {"duration": 0.01, "generator": "noise", "seed": 7}
    samples = m.render(settings)
    result = m.manifest(settings, samples)
    assert set(result) == {
        "schema_version",
        "config",
        "frames",
        "channels",
        "sha256",
        "peak",
        "rms",
    }
    assert result["frames"] == 160 and result["channels"] == 1
    assert result["sha256"] == m.audio_digest(samples, 16000)
    assert result == m.manifest(settings, samples)


def test_manifest_encode_example():
    assert m.manifest_encode(m.manifest({"duration": 0}, [])).endswith("\n")


def test_manifest_encode_boundaries():
    value = m.manifest({"duration": 0}, [])
    for key, invalid in [
        ("frames", True),
        ("channels", 2),
        ("schema_version", True),
        ("sha256", "x" * 64),
        ("peak", np.nan),
        ("rms", -1),
    ]:
        with pytest.raises(ValueError):
            m.manifest_encode(dict(value, **{key: invalid}))


def test_manifest_encode_invariants():
    import json

    value = m.manifest({"duration": 0}, [])
    assert json.loads(m.manifest_encode(value)) == value
    assert m.manifest_encode(dict(reversed(list(value.items())))) == m.manifest_encode(value)
    for key in value:
        altered = dict(value)
        altered.pop(key)
        with pytest.raises(ValueError):
            m.manifest_encode(altered)
    with pytest.raises(ValueError):
        m.manifest_encode(dict(value, unexpected=1))


def test_manifest_decode_example():
    value = m.manifest({"duration": 0}, [])
    assert m.manifest_decode(m.manifest_encode(value)) == value


def test_manifest_decode_boundaries():
    for text in ["", "null", "[]", "{}", '{"x":1,"x":2}', '{"x":NaN}', "x" * 100001]:
        with pytest.raises(ValueError):
            m.manifest_decode(text)


def test_manifest_decode_invariants():
    for generator in ["sine", "noise", "chirp"]:
        settings = {"generator": generator, "duration": 0.001}
        value = m.manifest(settings, m.render(settings))
        encoded = m.manifest_encode(value)
        assert m.manifest_encode(m.manifest_decode(encoded)) == encoded


def test_generate_run_example():
    result = m.generate_run(
        [{"duration": 0.001}, {"duration": 0.001, "seed": 1, "generator": "noise"}]
    )
    assert len(result) == 2 and all(x["audio"].shape == (16, 1) for x in result)


def test_generate_run_boundaries():
    for configurations in [[], [None], [{"unknown": 1}], [{}] * 1001]:
        with pytest.raises(ValueError):
            m.generate_run(configurations)


def test_generate_run_invariants():
    settings = m.sweep({"duration": 0.001, "generator": "noise"}, [1, 2], [440])
    left, right = m.generate_run(settings), m.generate_run(settings)
    for a, b in zip(left, right):
        assert a["manifest"] == b["manifest"]
        np.testing.assert_array_equal(a["audio"], b["audio"])
    assert left[0]["manifest"]["sha256"] != left[1]["manifest"]["sha256"]


def test_write_run_example():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary:
        files = m.write_run(str(Path(temporary) / "run"), m.generate_run([{"duration": 0.001}]))
        assert len(files) == 1 and Path(files[0]).read_bytes()[:4] == b"RIFF"
        assert (Path(temporary) / "run" / "manifest.json").is_file()


def test_write_run_boundaries():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary:
        with pytest.raises(FileExistsError):
            m.write_run(temporary, m.generate_run([{"duration": 0}]))
        result = m.generate_run([{"duration": 0.001}])
        result[0]["audio"][0] = 99
        with pytest.raises(ValueError):
            m.write_run(str(Path(temporary) / "bad"), result)
        assert not (Path(temporary) / "bad").exists()
