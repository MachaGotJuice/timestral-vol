import numpy as np
import pandas as pd
import pytest

from src.data import realized_forward, target_30d
from src.finetune import (
    CONTEXT_LEN,
    HORIZON,
    TRAIN_END,
    VAL_END,
    VAL_START,
    WindowSet,
    split_windows,
    ticker_windows,
)


def _daily_var(start="2005-01-03", end="2023-06-30", seed=0):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    log_v = -9 + 0.3 * np.cumsum(rng.standard_normal(len(dates))) / np.sqrt(len(dates))
    return pd.Series(np.exp(log_v + 0.5 * rng.standard_normal(len(dates))), index=dates)


def test_window_layout_matches_target_and_realized():
    dv = _daily_var()
    y = target_30d(dv, window=HORIZON)
    w = ticker_windows(dv, "X")

    for i in [0, 500, len(w) - 1]:
        s = w.origin_dates[i]
        pos = y.index.get_loc(s)
        np.testing.assert_allclose(w.contexts[i], y.iloc[pos - CONTEXT_LEN + 1 : pos + 1].values, rtol=1e-6)
        np.testing.assert_allclose(w.targets[i], y.iloc[pos + 1 : pos + 1 + HORIZON].values, rtol=1e-6)
        assert w.target_end_dates[i] == y.index[pos + HORIZON]
        assert w.realized_var[i] == pytest.approx(realized_forward(dv, s, horizon=HORIZON), rel=1e-12)


def test_split_boundaries():
    windows = WindowSet.concat([ticker_windows(_daily_var(seed=k), t) for k, t in enumerate(["A", "B"])])
    train, val = split_windows(windows, val_step=5)

    assert train.target_end_dates.max() <= TRAIN_END
    assert val.origin_dates.min() >= VAL_START
    assert val.target_end_dates.max() <= VAL_END
    # the training set uses every origin, right up to the boundary
    last_train = windows.subset(windows.target_end_dates <= TRAIN_END).origin_dates.max()
    assert train.origin_dates.max() == last_train
    # validation is every 5th origin, per ticker
    for t in ["A", "B"]:
        d = val.origin_dates[val.tickers == t]
        all_d = windows.origin_dates[(windows.tickers == t) & (windows.origin_dates >= VAL_START)
                                     & (windows.target_end_dates <= VAL_END)]
        assert list(d) == list(all_d[::5])
