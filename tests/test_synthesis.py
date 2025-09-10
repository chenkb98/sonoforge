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


def test_noise_example():
    assert m.noise(0.01, 16000, 7).shape == (160, 1)


def test_noise_boundaries():
    for seed in [-1, True, 1.5, 2**63]:
        with pytest.raises(ValueError):
            m.noise(0.01, seed=seed)
    with pytest.raises(ValueError):
        m.noise(0.01, amplitude=np.nan)
    assert m.noise(0).shape == (0, 1)


def test_noise_invariants():
    np.testing.assert_array_equal(m.noise(0.01, seed=17), m.noise(0.01, seed=17))
    assert not np.array_equal(m.noise(0.01, seed=17), m.noise(0.01, seed=18))
    assert np.max(np.abs(m.noise(1, seed=5, amplitude=0.2))) <= 0.2
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    m.noise(0.01)
    assert np.random.random() == expected


def test_chirp_example():
    np.testing.assert_allclose(m.chirp(1, 1, 1, 8), m.sine(1, 1, 8))


def test_chirp_boundaries():
    assert m.chirp(1, 2, 0, 8).shape == (0, 1)
    with pytest.raises(ValueError):
        m.chirp(-1, 2, 1, 8)
    with pytest.raises(ValueError):
        m.chirp(1, 4, 1, 8)
    with pytest.raises(ValueError):
        m.chirp(1, 2, -1, 8)


def test_chirp_invariants():
    for start, end in [(100, 1000), (1000, 100)]:
        x = m.chirp(start, end, 0.1, 8000, 0.25)
        assert x.shape == (800, 1) and np.max(np.abs(x)) <= 0.25
        assert x[0, 0] == 0
        np.testing.assert_array_equal(m.chirp(start, end, 0.1, 8000, 0.25), x)


def test_harmonics_example():
    np.testing.assert_allclose(m.harmonics(1, [1], 1, 8), m.sine(1, 1, 8, 1))


def test_harmonics_boundaries():
    for weights in [[], [np.nan], [[1]], [np.inf]]:
        with pytest.raises(ValueError):
            m.harmonics(1, weights, 1, 8)
    with pytest.raises(ValueError):
        m.harmonics(1, [1, 1, 1, 1], 1, 8)
    with pytest.raises(ValueError):
        m.harmonics(0, [1], 1, 8)


def test_harmonics_invariants():
    x = m.harmonics(1, [1, 0.5, -0.25], 1, 16)
    assert np.max(np.abs(x)) <= 1
    np.testing.assert_allclose(m.harmonics(1, [2, 1, -0.5], 1, 16), x)
    np.testing.assert_array_equal(m.harmonics(1, [0, 0], 1, 16), np.zeros((16, 1)))


def test_adsr_example():
    np.testing.assert_allclose(m.adsr(2, 2, 2, 2, 0.5), [0, 0.5, 1, 0.75, 0.5, 0.5, 0.5, 0])


def test_adsr_boundaries():
    assert len(m.adsr(0, 0, 0, 0)) == 0
    for lengths in [(-1, 0, 0, 0), (0, 1.5, 0, 0), (0, 0, True, 0)]:
        with pytest.raises(ValueError):
            m.adsr(*lengths)
    with pytest.raises(ValueError):
        m.adsr(1, 1, 1, 1, 2)


def test_adsr_invariants():
    for attack in range(3):
        for release in range(3):
            env = m.adsr(attack, 2, 3, release, 0.4)
            assert len(env) == attack + release + 5
            assert np.all((env >= 0) & (env <= 1))
            if release:
                assert env[-1] == 0


def test_apply_envelope_example():
    np.testing.assert_allclose(
        m.apply_envelope([[1, -1], [1, -1]], [0, 0.5]), [[0, 0], [0.5, -0.5]]
    )


def test_apply_envelope_boundaries():
    for envelope in [[1], [-1, 1], [np.nan, 1], [[1], [1]]]:
        with pytest.raises(ValueError):
            m.apply_envelope([1, 2], envelope)
    assert m.apply_envelope([], []).shape == (0, 1)


def test_apply_envelope_invariants():
    x = np.arange(8).reshape(4, 2)
    np.testing.assert_array_equal(m.apply_envelope(x, np.ones(4)), x)
    np.testing.assert_array_equal(m.apply_envelope(x, np.zeros(4)), np.zeros((4, 2)))
    a, b = np.array([1, 0.5, 0.2, 0]), np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        m.apply_envelope(m.apply_envelope(x, a), b), m.apply_envelope(x, a * b)
    )
