import numpy as np
import pandas as pd

from src.data import garman_klass, split_by_date, target_30d


def _synthetic_ohlc(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + rng.standard_normal(n).cumsum()
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=dates)


def test_garman_klass_is_positive():
    df = _synthetic_ohlc()
    gk = garman_klass(df)
    assert (gk > 0).all()


def test_target_30d_no_lookahead():
    df = _synthetic_ohlc(n=200)
    gk = garman_klass(df)
    y1 = target_30d(gk, window=30)

    gk_modified = gk.copy()
    gk_modified.iloc[101:] = gk_modified.iloc[101:] * 1000.0
    y2 = target_30d(gk_modified, window=30)

    cutoff = df.index[100]
    idx = y1.index[y1.index <= cutoff]
    pd.testing.assert_series_equal(y1.loc[idx], y2.loc[idx])


def test_split_by_date_chronological_and_non_overlapping():
    dates = pd.bdate_range("2015-01-01", "2023-12-31")
    series = pd.Series(np.arange(len(dates)), index=dates)
    splits = split_by_date(series)

    assert splits["train"].index.max() <= pd.Timestamp("2018-12-31")
    assert splits["val"].index.min() >= pd.Timestamp("2019-01-01")
    assert splits["val"].index.max() <= pd.Timestamp("2021-12-31")
    assert splits["test"].index.min() >= pd.Timestamp("2022-01-01")

    all_idx = splits["train"].index.append(splits["val"].index).append(splits["test"].index)
    assert len(all_idx) == len(set(all_idx))
    assert len(all_idx) == len(series)
