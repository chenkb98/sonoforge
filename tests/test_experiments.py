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
