"""Data loading and target construction for the timestral-vol project.

No torch/timestral imports live here on purpose: this module must be
importable and testable in a plain, torch-free local environment.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "AAPL", "MSFT", "JPM", "XOM"]
DEFAULT_START = "2005-01-01"


def download_ohlc(ticker, start=DEFAULT_START, end=None, cache_dir="cache"):
    """Download daily OHLC data for `ticker` via yfinance, with CSV caching.

    Uses auto_adjust=True (so Open/High/Low/Close are split/dividend
    adjusted), flattens the MultiIndex columns yfinance now returns for
    single tickers, caches the result as a CSV under `cache_dir`, and
    drops any rows containing NaNs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_{start}_{end or 'latest'}.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    df.to_csv(cache_path)
    return df


def garman_klass(df, floor=1e-10):
    """Garman-Klass daily variance estimator.

    var_t = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

    Clipped at `floor` to keep the series strictly positive (needed for
    log-targets and QLIKE downstream).
    """
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return gk.clip(lower=floor)


def log_returns(df):
    """Close-to-close log returns."""
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def target_30d(daily_var, window=30):
    """Log of the trailing `window`-day mean of a daily variance series.

    y_t = log( mean(daily_var[t-window+1 .. t]) )

    This only ever looks backward from t, so it introduces no lookahead.
    The forecasting task built on top of this target is: given a context
    ending at time t, predict y_{t+window}. Since y_{t+window} is the
    trailing average over days [t+1 .. t+window], predicting it is
    exactly predicting the average realized variance over the next
    `window` days -- i.e. a 30-day-ahead volatility forecast.
    """
    rolling_mean = daily_var.rolling(window=window, min_periods=window).mean()
    return np.log(rolling_mean).dropna()


def split_by_date(series):
    """Chronological train/val/test split.

    train: <= 2018-12-31
    val:   2019-01-01 .. 2021-12-31
    test:  >= 2022-01-01
    """
    train = series[series.index <= "2018-12-31"]
    val = series[(series.index >= "2019-01-01") & (series.index <= "2021-12-31")]
    test = series[series.index >= "2022-01-01"]
    return {"train": train, "val": val, "test": test}
