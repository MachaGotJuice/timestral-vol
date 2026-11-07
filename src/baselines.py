"""Classical volatility forecasting baselines: GARCH and HAR.

Each function makes ONE forecast at ONE origin t, using only data dated
<= t. The rolling/expanding evaluation loop over tickers and origins
lives in notebooks/01_data_and_baselines.ipynb, not here.
"""
import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch_forecast(returns, daily_var, t, window=1000, horizon=30, return_fit=False):
    """Fit a zero-mean GARCH(1,1)-t on the last `window` returns ending at
    t (inclusive) and forecast the mean daily variance over the next
    `horizon` days.

    Refit from scratch every call -- this function is meant to be called
    once per forecast origin in a rolling-window evaluation.

    GARCH is fit on close-to-close returns, so its forecast targets
    close-to-close variance. The evaluation target is Garman-Klass
    variance instead, which also captures the overnight gap and lives on
    a different scale. We correct for this with a simple multiplicative
    scale factor estimated on the same fitting window:

        c_t = mean(Garman-Klass variance over the window)
              / mean(close-to-close r^2 over the window)

    applied to the raw GARCH forecast. This keeps GARCH's dynamics
    (volatility clustering, fat tails) while matching its level to the
    Garman-Klass scale that every model is evaluated on.

    Parameters
    ----------
    returns : pd.Series
        Close-to-close log returns (decimal, not percent), DatetimeIndex.
    daily_var : pd.Series
        Garman-Klass daily variance, same index convention as `returns`.
    t : pd.Timestamp
        Forecast origin. Only data with index <= t is used.
    window : int
        Number of trailing observations used to fit GARCH and the scale
        factor.
    horizon : int
        Number of days ahead to average the variance forecast over.
    return_fit : bool
        If True, also return the fitted arch results object (for
        diagnostics, e.g. `res.summary()`).

    Returns
    -------
    raw_var : float
        Unscaled GARCH forecast, close-to-close variance units.
    scaled_var : float
        raw_var * c_t -- the official baseline, comparable to
        Garman-Klass targets.
    res : arch.univariate.base.ARCHModelResult, optional
        Only if return_fit=True.
    """
    common_idx = returns.index.intersection(daily_var.index)
    window_idx = common_idx[common_idx <= t][-window:]
    assert window_idx.max() <= t, "GARCH window must not use data after t"

    r = returns.loc[window_idx]
    gk = daily_var.loc[window_idx]

    # arch's optimizer is better conditioned on percentage returns.
    am = arch_model(r * 100.0, mean="Zero", vol="Garch", p=1, q=1, dist="t")
    res = am.fit(disp="off")

    fc = res.forecast(horizon=horizon, reindex=False)
    mean_var_pct2 = fc.variance.values[-1].mean()
    raw_var = float(mean_var_pct2 / 100.0**2)

    c_t = float(gk.mean() / (r**2).mean())
    scaled_var = raw_var * c_t

    if return_fit:
        return raw_var, scaled_var, res
    return raw_var, scaled_var


def _har_features(daily_var):
    """Trailing log-mean HAR regressors: daily, weekly (5d), monthly (22d).

    Row s only uses daily_var up to and including s -- no lookahead.
    """
    daily = np.log(daily_var)
    weekly = np.log(daily_var.rolling(5, min_periods=5).mean())
    monthly = np.log(daily_var.rolling(22, min_periods=22).mean())
    return pd.DataFrame({"daily": daily, "weekly": weekly, "monthly": monthly})


def _har_targets(daily_var, horizon):
    """Vectorized log(mean(daily_var[s+1 .. s+horizon])) for every s.

    Same quantity as data.realized_forward(daily_var, s, horizon), just
    computed for every s at once instead of one Timestamp at a time.
    """
    forward_mean = daily_var.rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    return np.log(forward_mean)


def fit_har_forecast(daily_var, t, horizon=30, return_fit=False):
    """Fit a direct, log-scale HAR-RV model and forecast at origin t.

    Features at time s (all using only data up to and including s):
        daily   = log GK[s]
        weekly  = log mean(GK[s-4 .. s])
        monthly = log mean(GK[s-21 .. s])
    Target at time s:
        target  = log mean(GK[s+1 .. s+horizon])

    Trained by OLS on an expanding window: every s from the start of the
    data such that s + horizon <= t, i.e. every training target is fully
    observed using only data up to t (no lookahead). The forecast for t
    applies a log-normal bias correction using the in-sample residual
    variance: forecast = exp(prediction + 0.5 * residual_variance).

    Parameters
    ----------
    daily_var : pd.Series
        Garman-Klass daily variance, DatetimeIndex.
    t : pd.Timestamp
        Forecast origin.
    horizon : int
        Forecast horizon in trading days.
    return_fit : bool
        If True, also return a dict with the fitted coefficients,
        in-sample residual variance, and training set size.

    Returns
    -------
    forecast_var : float
    fit_info : dict, optional
        Only if return_fit=True.
    """
    features = _har_features(daily_var)
    targets = _har_targets(daily_var, horizon)

    dataset = features.copy()
    dataset["target"] = targets
    dataset = dataset.dropna()

    t_pos = daily_var.index.get_loc(t)
    s_pos = daily_var.index.get_indexer(dataset.index)
    train_mask = s_pos + horizon <= t_pos

    train = dataset.loc[train_mask]
    assert len(train) > 0, "not enough history to fit HAR at this origin"
    assert (daily_var.index.get_indexer(train.index) + horizon <= t_pos).all(), (
        "HAR training must never use a target window ending after t"
    )

    feature_cols = ["daily", "weekly", "monthly"]
    X_train = np.column_stack([np.ones(len(train)), train[feature_cols].values])
    y_train = train["target"].values

    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    resid = y_train - X_train @ beta
    resid_var = float(np.var(resid, ddof=X_train.shape[1]))

    x_t = features.loc[t]
    assert x_t.notna().all(), "missing HAR features at origin t"
    x_t_vec = np.array([1.0, x_t["daily"], x_t["weekly"], x_t["monthly"]])
    y_hat = float(x_t_vec @ beta)

    forecast_var = float(np.exp(y_hat + 0.5 * resid_var))

    if return_fit:
        fit_info = {
            "beta": beta,
            "feature_names": ["intercept", *feature_cols],
            "resid_var": resid_var,
            "n_train": len(train),
        }
        return forecast_var, fit_info
    return forecast_var
