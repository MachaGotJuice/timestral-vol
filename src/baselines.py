"""Classical volatility forecasting baselines: GARCH and HAR.

Stubs only for now -- implemented in notebooks/01_data_and_baselines.ipynb
once this scaffold is in place.
"""


def fit_garch_forecast(returns, horizon=30):
    """Fit a GARCH(1,1) model on `returns` and forecast variance.

    TODO:
    - Fit arch.arch_model(returns, vol="Garch", p=1, q=1) on the training
      window.
    - Forecast `horizon` steps ahead and aggregate to a 30-day average
      variance forecast comparable to target_30d().
    - Return a pandas Series aligned with the dates being forecast.
    """
    raise NotImplementedError


def fit_har_forecast(daily_var, horizon=30):
    """Fit a HAR-RV (Heterogeneous AutoRegressive) model and forecast variance.

    TODO:
    - Build daily/weekly/monthly realized-variance regressors from
      `daily_var`.
    - Fit an OLS (or similar) HAR model.
    - Forecast `horizon` steps ahead and aggregate to a 30-day average
      variance forecast comparable to target_30d().
    - Return a pandas Series aligned with the dates being forecast.
    """
    raise NotImplementedError
