"""Comparison tables shared by notebooks 02 and 03 (pandas only, no torch).

All functions take a tidy forecasts frame with one row per
(ticker, origin_date, model) and columns forecast_var, realized_var,
and optionally q10 / q90.
"""
import numpy as np
import pandas as pd

from src.metrics import dm_test, qlike

KEY = ["ticker", "origin_date"]


def load_forecasts(*paths):
    """Read and stack forecast CSVs; realized_var must be identical for each (ticker, origin_date)."""
    frames = [pd.read_csv(p, parse_dates=["origin_date"]) for p in paths]
    return combine_forecasts(*frames)


def combine_forecasts(*frames):
    """Stack forecast frames, asserting they share realized_var on every (ticker, origin_date)."""
    ref = frames[0].drop_duplicates(KEY).set_index(KEY)["realized_var"]
    for f in frames[1:]:
        other = f.drop_duplicates(KEY).set_index(KEY)["realized_var"]
        assert set(other.index) == set(ref.index), "frames cover different (ticker, origin_date) pairs"
        np.testing.assert_allclose(other.loc[ref.index].values, ref.values, rtol=1e-10)
    out = pd.concat(frames, ignore_index=True)
    assert not out.duplicated(KEY + ["model"]).any(), "a model appears in more than one frame"
    return out


def qlike_losses(df):
    """Per-row QLIKE loss r - log r - 1, r = realized / forecast."""
    r = df["realized_var"].values / df["forecast_var"].values
    return pd.Series(r - np.log(r) - 1, index=df.index)


def qlike_tables(df, models):
    """(pooled, per_ticker, per_year) QLIKE, one column per model, in `models` order."""
    sub = df[df["model"].isin(models)].assign(loss=lambda d: qlike_losses(d), year=lambda d: d["origin_date"].dt.year)
    pooled = sub.groupby("model")["loss"].mean().reindex(models).rename("qlike")
    per_ticker = sub.pivot_table(index="ticker", columns="model", values="loss", aggfunc="mean")
    per_ticker = per_ticker.reindex(index=pd.unique(sub["ticker"]), columns=models)
    per_year = sub.pivot_table(index="year", columns="model", values="loss", aggfunc="mean")[models]
    # sanity: the groupby means are the same number src.metrics.qlike gives
    m0 = sub[sub["model"] == models[0]]
    assert np.isclose(pooled.iloc[0], qlike(m0["realized_var"], m0["forecast_var"]))
    return pooled, per_ticker, per_year


def _aligned_losses(df, model):
    sub = df[df["model"] == model].assign(loss=lambda d: qlike_losses(d))
    return sub.set_index(KEY)["loss"].sort_index()


def dm_table(df, model, ref, lag=5):
    """Diebold-Mariano of `model` vs `ref` on QLIKE losses, per ticker and pooled.

    Pooled = one DM test on the cross-ticker average loss per origin date.
    mean_diff < 0 means `model` has the lower loss.
    """
    a, b = _aligned_losses(df, model), _aligned_losses(df, ref)
    assert a.index.equals(b.index), f"{model} and {ref} are not on the same (ticker, origin_date) set"
    rows = []
    for ticker in pd.unique(df["ticker"]):
        mean_diff, stat, p = dm_test(a.loc[ticker].values, b.loc[ticker].values, lag=lag)
        rows.append({"ticker": ticker, "mean_diff": mean_diff, "dm_stat": stat, "p_value": p})
    per_date = a.groupby(level="origin_date").count()
    assert per_date.nunique() == 1, "pooled DM needs every ticker on every origin date"
    mean_diff, stat, p = dm_test(a.groupby(level="origin_date").mean().values,
                                 b.groupby(level="origin_date").mean().values, lag=lag)
    rows.append({"ticker": "pooled", "mean_diff": mean_diff, "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows).set_index("ticker")


def calibration_table(df, models):
    """10-90% band coverage (target 0.80) and mean log bias log(forecast / realized)."""
    rows = []
    for model in models:
        sub = df[df["model"] == model]
        row = {"model": model, "log_bias": float(np.mean(np.log(sub["forecast_var"] / sub["realized_var"])))}
        if {"q10", "q90"} <= set(sub.columns) and sub["q10"].notna().all():
            row["coverage_10_90"] = float(((sub["realized_var"] >= sub["q10"]) & (sub["realized_var"] <= sub["q90"])).mean())
        else:
            row["coverage_10_90"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
