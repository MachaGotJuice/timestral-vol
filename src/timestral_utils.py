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


def nearest_quantile(values, levels, level):
    """Value at the sampled quantile level closest to `level`.

    `values` has shape (..., Q) and `levels` shape (Q,) (e.g.
    forecast.quantile_levels). Used for e.g. the median (level=0.5) or a
    10/90% band edge, without ever assuming the levels are evenly spaced.
    """
    idx = int(np.argmin(np.abs(np.asarray(levels) - level)))
    return values[..., idx]


def quantile_mean(values, levels):
    """E[X] from samples of X's quantile function, trapezoidal + flat tails.

    `values[..., i]` is the quantile of X at probability `levels[i]`
    (levels ascending, e.g. forecast.quantile_levels). The mean is
    E[X] = integral_0^1 Q(p) dp, approximated as:
      - the trapezoidal rule over the sampled region [levels[0], levels[-1]]
      - plus flat-tail terms levels[0] * values[..., 0] and
        (1 - levels[-1]) * values[..., -1] for the unsampled tails,
        assuming Q(p) is roughly constant beyond the sampled extremes.

    This is an approximation (exact only where that flat-tail assumption
    holds), not a substitute for the model's true expectation.
    """
    levels = np.asarray(levels)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz
    inner = trapezoid(values, levels, axis=-1)
    lower_tail = levels[0] * values[..., 0]
    upper_tail = (1 - levels[-1]) * values[..., -1]
    return lower_tail + inner + upper_tail


def differentiable_quantiles(forecaster, contexts, horizon):
    """Quantile forecasts that keep the autograd graph, for fine-tuning.

    `forecaster.forecast()` is wrapped in @torch.inference_mode(), so it
    gives no gradients. This calls the private
    `forecaster._forecast_raw(inputs, mask, horizon)` under
    `torch.enable_grad()` instead -- the same single pass `forecast()`
    makes, without symmetric averaging and without non-negative clipping
    (tests/test_differentiable.py checks the two agree). It is a private
    API: keep the timestral commit pinned in requirements.txt.

    `_forecast_raw` calls `model.eval()`, so dropout (if any) is off even
    while training; that is fine for full fine-tuning here.

    Parameters
    ----------
    forecaster : TimestralForecaster
    contexts : torch.Tensor, shape (B, T)
        Float context series on the forecaster's device. NaN/inf mark
        missing values, exactly as in `forecast()`.
    horizon : int

    Returns
    -------
    torch.Tensor, shape (B, 1, H, Q)
        Quantiles in data units, differentiable w.r.t. forecaster.model.
    """
    import torch

    inputs = contexts.to(device=forecaster.device, dtype=torch.float32)[:, None, :]
    mask = torch.isfinite(inputs)
    inputs = torch.where(mask, inputs, torch.zeros_like(inputs))  # as forecast() does
    with torch.enable_grad():
        return forecaster._forecast_raw(inputs, mask, horizon)


def load_finetuned(checkpoint_path, size="5M", device=None):
    """Timestral forecaster with weights from a fine-tuning checkpoint.

    The checkpoint is a dict saved by `src.finetune.train` with the
    model's state_dict under "model".
    """
    import torch

    forecaster = load_model(size, device=device)
    state = torch.load(checkpoint_path, map_location=forecaster.device)
    forecaster.model.load_state_dict(state["model"])
    forecaster.model.eval()
    return forecaster
