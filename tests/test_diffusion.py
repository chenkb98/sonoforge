import numpy as np
import pytest
from sonoforge import diffusion as m


def test_linear_schedule_example():
    np.testing.assert_allclose(m.linear_schedule(3, 0.1, 0.3), [0.1, 0.2, 0.3])
