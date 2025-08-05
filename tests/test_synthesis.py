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


def test_timebase_invariants():
    for rate in [8000, 16000, 24000]:
        clock = m.timebase(0.01, rate)
        assert len(clock) == rate // 100
        assert clock[0] == 0 and clock[-1] < 0.01
        np.testing.assert_allclose(np.diff(clock), 1 / rate)


def test_sine_example():
    np.testing.assert_allclose(m.sine(1, 1, 4, 1).ravel(), [0, 1, 0, -1], atol=1e-15)
