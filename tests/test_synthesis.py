import numpy as np
import pytest
from sonoforge import synthesis as m


def test_timebase_example():
    np.testing.assert_array_equal(m.timebase(1, 4), [0, 0.25, 0.5, 0.75])


def test_timebase_boundaries():
    for seconds in [-1, np.nan, np.inf, 1e10]:
        with pytest.raises(ValueError):
            m.timebase(seconds, 16000)
    assert len(m.timebase(0, 16000)) == 0
    with pytest.raises(ValueError):
        m.timebase(1, 0)
