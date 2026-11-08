import numpy as np
import pandas as pd
import pytest

from src.evaluation import calibration_table, combine_forecasts, dm_table, qlike_tables
from src.metrics import dm_test, qlike


def _frame(model, seed, n_dates=40, tickers=("A", "B")):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates, freq="5B").append(pd.bdate_range("2023-01-02", periods=n_dates, freq="5B"))
    rows = []
    for t in tickers:
        r = np.exp(rng.standard_normal(len(dates)) * 0.0 - 9 + np.sin(np.arange(len(dates)) / 7))
        f = r * np.exp(0.3 * np.random.default_rng(seed + 100).standard_normal(len(dates)))
        rows.append(pd.DataFrame({"ticker": t, "origin_date": dates, "model": model, "forecast_var": f,
                                  "realized_var": r, "q10": f * 0.7, "q90": f * 1.4}))
    return pd.concat(rows, ignore_index=True)


def test_combine_rejects_mismatched_realized():
    a, b = _frame("m1", 0), _frame("m2", 1)
    combine_forecasts(a, b)
    b.loc[0, "realized_var"] *= 1.01
    with pytest.raises(AssertionError):
        combine_forecasts(a, b)


def test_qlike_tables_match_metrics():
    df = combine_forecasts(_frame("m1", 0), _frame("m2", 1))
    pooled, per_ticker, per_year = qlike_tables(df, ["m2", "m1"])
    assert list(pooled.index) == ["m2", "m1"]
    sub = df[(df.model == "m1") & (df.ticker == "B")]
    assert per_ticker.loc["B", "m1"] == pytest.approx(qlike(sub.realized_var, sub.forecast_var))
    sub = df[(df.model == "m2") & (df.origin_date.dt.year == 2023)]
    assert per_year.loc[2023, "m2"] == pytest.approx(qlike(sub.realized_var, sub.forecast_var))


def test_dm_table_pooled_averages_across_tickers():
    df = combine_forecasts(_frame("m1", 0), _frame("m2", 1))
    table = dm_table(df, "m1", "m2", lag=5)
    assert list(table.index) == ["A", "B", "pooled"]

    def losses(m):
        s = df[df.model == m].sort_values(["origin_date", "ticker"])
        r = s.realized_var.values / s.forecast_var.values
        return pd.Series(r - np.log(r) - 1).groupby(s.origin_date.values).mean().values

    _, stat, p = dm_test(losses("m1"), losses("m2"), lag=5)
    assert table.loc["pooled", "p_value"] == pytest.approx(p)


def test_calibration_table():
    df = _frame("m1", 0)
    cal = calibration_table(df, ["m1"])
    inside = (df.realized_var >= df.q10) & (df.realized_var <= df.q90)
    assert cal.loc["m1", "coverage_10_90"] == pytest.approx(inside.mean())
    assert cal.loc["m1", "log_bias"] == pytest.approx(np.mean(np.log(df.forecast_var / df.realized_var)))
