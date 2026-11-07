import numpy as np
import pytest

from src.timestral_utils import nearest_quantile, quantile_mean


def test_nearest_quantile_picks_closest_level():
    levels = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    assert nearest_quantile(values, levels, 0.5) == 30.0
    assert nearest_quantile(values, levels, 0.1) == 20.0  # closest to 0.05


def test_quantile_mean_matches_formula():
    levels = np.array([0.01, 0.5, 0.99])
    values = np.array([1.0, 2.0, 5.0])

    expected = levels[0] * values[0] + np.trapezoid(values, levels) + (1 - levels[-1]) * values[-1]
    assert quantile_mean(values, levels) == pytest.approx(expected)


def test_quantile_mean_batched():
    levels = np.array([0.01, 0.5, 0.99])
    values = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, 10.0]])

    result = quantile_mean(values, levels)
    assert result.shape == (2,)

    expected0 = levels[0] * 1.0 + np.trapezoid([1.0, 2.0, 5.0], levels) + (1 - levels[-1]) * 5.0
    assert result[0] == pytest.approx(expected0)
