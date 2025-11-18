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


def test_forward_noise_boundaries():
    for alpha in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.forward_noise([1], [0], alpha)
    with pytest.raises(ValueError):
        m.forward_noise([1], [0, 1], 0.5)
    with pytest.raises(ValueError):
        m.forward_noise([np.nan], [0], 0.5)


def test_forward_noise_invariants():
    x, epsilon = np.array([1, -1]), np.array([0.3, 0.7])
    np.testing.assert_array_equal(m.forward_noise(x, epsilon, 1), x)
    np.testing.assert_array_equal(m.forward_noise(x, epsilon, 0), epsilon)
    np.testing.assert_allclose(m.forward_noise(x, epsilon, 0.3), m.forward_noise(epsilon, x, 0.7))


def test_predict_clean_example():
    np.testing.assert_allclose(m.predict_clean([0.5, np.sqrt(0.75)], [0, 1], 0.25), [1, 0])


def test_predict_clean_boundaries():
    for alpha in [0, -1, 1e-20, np.nan]:
        with pytest.raises(ValueError):
            m.predict_clean([1], [0], alpha)
    with pytest.raises(ValueError):
        m.predict_clean([1], [0, 0], 0.5)
    np.testing.assert_array_equal(m.predict_clean([1, 2], [9, 9], 1), [1, 2])


def test_predict_clean_invariants():
    rng = np.random.default_rng(3)
    clean, epsilon = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
    for alpha in [0.001, 0.1, 0.5, 0.99, 1]:
        np.testing.assert_allclose(
            m.predict_clean(m.forward_noise(clean, epsilon, alpha), epsilon, alpha),
            clean,
            atol=1e-12,
        )


def test_ddim_step_example():
    np.testing.assert_allclose(m.ddim_step([0.5], [0], 0.25, 1), [1])


def test_ddim_step_boundaries():
    for previous in [0.1, 1.1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.ddim_step([1], [0], 0.25, previous)
    with pytest.raises(ValueError):
        m.ddim_step([1], [0], 0, 1)


def test_ddim_step_invariants():
    clean, epsilon = [1, -1, 0.2], [0.1, 0.3, -0.2]
    noisy = m.forward_noise(clean, epsilon, 0.2)
    np.testing.assert_allclose(m.ddim_step(noisy, epsilon, 0.2, 0.2), noisy)
    via = m.ddim_step(m.ddim_step(noisy, epsilon, 0.2, 0.6), epsilon, 0.6, 1)
    np.testing.assert_allclose(via, clean, atol=1e-14)


def test_initialize_model_example():
    model = m.initialize_model(4, 3)
    assert set(model) == {"weights", "bias"}
    assert model["weights"].shape == (5, 4) and model["bias"].shape == (4,)


def test_initialize_model_boundaries():
    for width in [0, 257, 1.5, True]:
        with pytest.raises(ValueError):
            m.initialize_model(width)
    for seed in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            m.initialize_model(seed=seed)


def test_initialize_model_invariants():
    a, b = m.initialize_model(8, 5), m.initialize_model(8, 5)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["weights"], m.initialize_model(8, 6)["weights"])
    a["weights"][0, 0] = 99
    assert b["weights"][0, 0] != 99


def test_predict_noise_example():
    model = {
        "weights": np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]),
        "bias": np.array([0.1, -0.1]),
    }
    np.testing.assert_allclose(m.predict_noise(model, [1, 2], 0.5), [[2.1, 3.4]])


def test_predict_noise_boundaries():
    model = m.initialize_model(2)
    for noisy, times in [([1], 0.5), ([1, 2], -1), ([1, np.nan], 0.5), ([[1, 2]], [0, 1])]:
        with pytest.raises(ValueError):
            m.predict_noise(model, noisy, times)
    with pytest.raises(ValueError):
        m.predict_noise({}, [1, 2], 0.5)


def test_predict_noise_invariants():
    model = m.initialize_model(3, 7)
    x = np.arange(12).reshape(4, 3)
    times = np.linspace(0, 1, 4)
    batch = m.predict_noise(model, x, times)
    for i in range(4):
        np.testing.assert_allclose(m.predict_noise(model, x[i], times[i])[0], batch[i])
    assert m.predict_noise(model, np.empty((0, 3)), []).shape == (0, 3)


def test_training_loss_example():
    model = {"weights": np.zeros((3, 2)), "bias": np.zeros(2)}
    assert m.training_loss(model, [[0, 0]], [0], [[1, 3]]) == 5
