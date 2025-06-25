import numpy as np
import pytest
from sonoforge import synthesis as m


def test_timebase_example():
    np.testing.assert_array_equal(m.timebase(1, 4), [0, 0.25, 0.5, 0.75])
