import numpy as np
import pytest
from sonoforge import audio as m


def test_as_audio_example():
    np.testing.assert_array_equal(m.as_audio([1, -1]), [[1], [-1]])
