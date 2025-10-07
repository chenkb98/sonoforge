import numpy as np
import pytest
from sonoforge import diffusion as m


def test_linear_schedule_example():
    np.testing.assert_allclose(m.linear_schedule(3, 0.1, 0.3), [0.1, 0.2, 0.3])


def test_linear_schedule_boundaries():
    for steps in [0, -1, 1.5, True, 10001]:
        with pytest.raises(ValueError):
            m.linear_schedule(steps)
    for start, end in [(0, 0.1), (0.2, 0.1), (0.1, 1), (np.nan, 0.1)]:
        with pytest.raises(ValueError):
            m.linear_schedule(5, start, end)


def test_linear_schedule_invariants():
    for steps in [1, 2, 17, 1000]:
        schedule = m.linear_schedule(steps)
        assert len(schedule) == steps
        assert np.all((schedule > 0) & (schedule < 1))
        assert np.all(np.diff(schedule) >= 0)
        assert schedule[0] == 0.0001


def test_cosine_schedule_example():
    np.testing.assert_allclose(m.cosine_schedule(2, 0), [0.5, 0.999], atol=1e-14)


def test_cosine_schedule_boundaries():
    for offset in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.cosine_schedule(5, offset)
    with pytest.raises(ValueError):
        m.cosine_schedule(0)
    assert m.cosine_schedule(1)[0] == 0.999
