import numpy as np
import pytest
from sonoforge import experiments as m


def test_config_example():
    value = m.config()
    assert value["generator"] == "sine" and value["sample_rate"] == 16000
