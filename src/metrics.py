"""Evaluation metrics for variance forecasts: QLIKE and Diebold-Mariano."""
import numpy as np
from scipy import stats


def qlike(realized_var, forecast_var):
    """QLIKE loss for variance forecasts.

    L = mean( r - log(r) - 1 ),  r = realized_var / forecast_var

    The standard "robust" loss for evaluating volatility forecasts
    (Patton, 2011): it is minimized in expectation by the true
    conditional variance even when the realized-variance proxy used in
    place of it is noisy, unlike plain MSE computed on variances.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    assert np.all(realized_var > 0), "realized_var must be strictly positive"
    assert np.all(forecast_var > 0), "forecast_var must be strictly positive"
    r = realized_var / forecast_var
    return float(np.mean(r - np.log(r) - 1))


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano test comparing two per-observation loss series.

    Tests H0: E[loss_a - loss_b] = 0, i.e. that models A and B have equal
    predictive accuracy under the loss function that produced loss_a/loss_b.

    The long-run variance of the loss differential is estimated with a
    Newey-West (Bartlett-kernel) estimator to account for serial
    correlation in the differential -- expected here since forecasts for
    a 30-day-ahead horizon use overlapping windows. As a rule of thumb,
    `lag` should be at least horizon - 1 (e.g. lag=29 for a 30-day-ahead
    horizon) to properly capture the induced autocorrelation.

    Returns
    -------
    mean_diff : float
        Mean of (loss_a - loss_b).
    dm_stat : float
        The Diebold-Mariano test statistic.
    p_value : float
        Two-sided p-value under a standard normal null.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    mean_diff = float(d.mean())

    d_centered = d - mean_diff
    gamma0 = np.dot(d_centered, d_centered) / n
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.dot(d_centered[k:], d_centered[:-k]) / n
        weight = 1 - k / (lag + 1)  # Bartlett kernel
        var += 2 * weight * gamma_k

    se = np.sqrt(var / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm_stat = mean_diff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return mean_diff, float(dm_stat), float(p_value)
