import numpy as np
import pandas as pd
import pytest

from src.baselines import fit_garch_forecast, fit_har_forecast


def _synthetic_returns_and_var(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(rng.standard_normal(n) * 0.01, index=dates)
    daily_var = pd.Series(np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6, index=dates)
    return returns, daily_var


def test_garch_forecast_uses_only_data_up_to_t():
    returns, daily_var = _synthetic_returns_and_var(n=300)
    t = returns.index[200]

    raw1, scaled1 = fit_garch_forecast(returns, daily_var, t, window=50, horizon=5)

    returns_changed = returns.copy()
    daily_var_changed = daily_var.copy()
    returns_changed.iloc[201:] *= 100.0
    daily_var_changed.iloc[201:] *= 100.0

    raw2, scaled2 = fit_garch_forecast(returns_changed, daily_var_changed, t, window=50, horizon=5)

    assert raw1 == pytest.approx(raw2)
    assert scaled1 == pytest.approx(scaled2)


def test_har_training_never_uses_target_ending_after_origin():
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    daily_var = pd.Series(np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6, index=dates)

    horizon = 10
    t = dates[100]
    t_pos = daily_var.index.get_loc(t)

    forecast1 = fit_har_forecast(daily_var, t, horizon=horizon)

    daily_var_changed = daily_var.copy()
    daily_var_changed.iloc[t_pos + 1 :] *= 1000.0
    forecast2 = fit_har_forecast(daily_var_changed, t, horizon=horizon)

    assert forecast1 == pytest.approx(forecast2)
