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
