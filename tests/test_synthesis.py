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


def test_sine_boundaries():
    for frequency in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.sine(frequency, 1, 4)
    with pytest.raises(ValueError):
        m.sine(1, 1, 4, 2)
    assert m.sine(1, 0, 4).shape == (0, 1)


def test_sine_invariants():
    x = m.sine(3, 1, 32, 0.7)
    assert x.shape == (32, 1) and np.max(np.abs(x)) <= 0.7
    np.testing.assert_allclose(m.sine(3, 1, 32, 0.7, np.pi), -x, atol=1e-14)
    np.testing.assert_array_equal(m.sine(3, 1, 32, 0), np.zeros((32, 1)))


def test_square_example():
    np.testing.assert_allclose(m.square(1, 1, 4, 1).ravel(), [1, 1, 1, -1], atol=1e-15)


def test_square_boundaries():
    for frequency in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.square(frequency, 1, 4)
    with pytest.raises(ValueError):
        m.square(1, 1, 4, -0.1)
    assert m.square(1, 0, 4).shape == (0, 1)


def test_square_invariants():
    x = m.square(1, 2, 16, 0.3)
    assert np.max(np.abs(x)) <= 0.3
    np.testing.assert_allclose(x[:16], x[16:])
    np.testing.assert_allclose(m.square(1, 2, 16, 0.6), x * 2)


def test_sawtooth_example():
    np.testing.assert_allclose(m.sawtooth(1, 1, 4, 1).ravel(), [-1, -0.5, 0, 0.5], atol=1e-15)


def test_sawtooth_boundaries():
    for frequency in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.sawtooth(frequency, 1, 4)
    with pytest.raises(ValueError):
        m.sawtooth(1, 1, 4, -0.1)
    assert m.sawtooth(1, 0, 4).shape == (0, 1)


def test_sawtooth_invariants():
    x = m.sawtooth(1, 2, 16, 0.3)
    assert np.max(np.abs(x)) <= 0.3
    np.testing.assert_allclose(x[:16], x[16:])
    np.testing.assert_allclose(m.sawtooth(1, 2, 16, 0.6), x * 2)


def test_triangle_example():
    np.testing.assert_allclose(m.triangle(1, 1, 4, 1).ravel(), [-1, 0, 1, 0], atol=1e-15)


def test_triangle_boundaries():
    for frequency in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.triangle(frequency, 1, 4)
    with pytest.raises(ValueError):
        m.triangle(1, 1, 4, -0.1)
    assert m.triangle(1, 0, 4).shape == (0, 1)


def test_triangle_invariants():
    x = m.triangle(1, 2, 16, 0.3)
    assert np.max(np.abs(x)) <= 0.3
    np.testing.assert_allclose(x[:16], x[16:])
    np.testing.assert_allclose(m.triangle(1, 2, 16, 0.6), x * 2)
