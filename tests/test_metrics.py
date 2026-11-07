import numpy as np

from src.metrics import dm_test, qlike


def test_qlike_zero_when_equal():
    realized = np.array([1.0, 2.0, 3.0])
    assert qlike(realized, realized) == 0.0


def test_qlike_positive_when_different():
    realized = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.5, 1.5, 4.0])
    assert qlike(realized, forecast) > 0


def test_dm_test_mean_diff_zero_for_identical_losses():
    rng = np.random.default_rng(0)
    loss = rng.random(100) + 0.1
    mean_diff, _, _ = dm_test(loss, loss, lag=5)
    assert mean_diff == 0.0


def test_dm_test_p_value_in_unit_interval_for_same_distribution():
    rng = np.random.default_rng(1)
    loss_a = rng.random(200) + 0.1
    loss_b = rng.random(200) + 0.1
    _, _, p_value = dm_test(loss_a, loss_b, lag=5)
    assert 0.0 <= p_value <= 1.0
