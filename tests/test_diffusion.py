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


def test_cosine_schedule_invariants():
    for steps in [2, 10, 100]:
        beta = m.cosine_schedule(steps)
        assert len(beta) == steps and np.all((beta > 0) & (beta < 1))
        alpha = np.cumprod(1 - beta)
        assert np.all(np.diff(alpha) < 0)
        np.testing.assert_array_equal(beta, m.cosine_schedule(steps))


def test_validate_schedule_example():
    np.testing.assert_array_equal(m.validate_schedule([0.1, 0.2]), [0.1, 0.2])


def test_validate_schedule_boundaries():
    for invalid in [[], [0], [1], [-1], [np.nan], [[0.1]], [".1"], [True], [0.999] * 10]:
        with pytest.raises(ValueError):
            m.validate_schedule(invalid)


def test_validate_schedule_invariants():
    x = np.array([0.1, 0.2, 0.3])
    y = m.validate_schedule(x)
    y[0] = 0.5
    assert x[0] == 0.1
    np.testing.assert_array_equal(m.validate_schedule(m.validate_schedule(x)), x)
    assert np.prod(1 - m.validate_schedule(m.linear_schedule(100))) > 0


def test_forward_noise_example():
    np.testing.assert_allclose(m.forward_noise([1, 0], [0, 1], 0.25), [0.5, np.sqrt(0.75)])
