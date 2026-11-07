"""Thin wrapper around the Timestral forecaster.

All torch / timestral imports are confined to this module (and done
lazily, inside each function) so the rest of the package (data,
baselines, metrics) stays importable in a torch-free local environment.
Only run this module's functions on Colab / a machine with torch>=2.4.
"""
import numpy as np


def load_model(size="5M", device=None):
    """Load a pretrained Timestral forecaster.

    Parameters
    ----------
    size : str
        Model size suffix, e.g. "5M" -> "mistralai/Timestral-5M".
    device : str or None
        "cuda" / "cpu". If None, uses cuda when available, else cpu.
    """
    import torch
    from timestral import TimestralForecaster

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device)


def forecast_quantiles(model, contexts, horizon, symmetric_averaging=True):
    """Run Timestral on a batch of contexts and extract quantile forecasts.

    Parameters
    ----------
    model : TimestralForecaster
    contexts : np.ndarray, shape (B, T)
        Batch of historical context series.
    horizon : int
        Number of steps to forecast ahead.
    symmetric_averaging : bool
        Passed through to model.forecast(). Keep True for plain
        inference use; see differentiable_quantiles() for why
        fine-tuning needs it off.

    Returns
    -------
    quantiles : np.ndarray, shape (B, H, Q)
    quantile_levels : np.ndarray, shape (Q,)
        The probability levels (e.g. 21 levels: 0.01, 0.05, 0.10, ...,
        0.95, 0.99), taken directly from forecast.quantile_levels --
        never synthesized with np.linspace, since the model's levels
        are not evenly spaced.
    """
    forecast = model.forecast(
        contexts,
        prediction_length=horizon,
        symmetric_averaging=symmetric_averaging,
    )
    quantiles = forecast.quantiles.detach().cpu().numpy()
    # (B, V, H, Q) -> (B, H, Q); assumes a single (univariate) variate.
    quantiles = quantiles[:, 0, :, :]
    quantile_levels = np.asarray(forecast.quantile_levels)
    return quantiles, quantile_levels


def differentiable_quantiles(model, contexts, horizon):
    """STUB -- not yet implemented. Needed for fine-tuning only.

    `model.forecast()` is decorated with @torch.inference_mode(), so it
    gives no gradients. For fine-tuning, call the internal
    `model._forecast_raw(inputs, mask, prediction_length)` under
    `torch.enable_grad()` instead. This is a private API: keep the
    pinned commit in requirements.txt, and add a test asserting this
    path is numerically equivalent to
    `model.forecast(..., symmetric_averaging=False)` before relying on
    it for fine-tuning.
    """
    raise NotImplementedError
