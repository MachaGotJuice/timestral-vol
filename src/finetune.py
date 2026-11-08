"""Fine-tuning Timestral on the 30-day log-variance target.

The data part (windows, splits) is plain numpy/pandas and importable
without torch, so it is tested locally. The training part imports torch
lazily, inside the functions that need it.

Task, identical to notebook 02's "smoothed" zero-shot variant:
    y = target_30d(GK daily variance)          (log of trailing 30-day mean)
    context = y[s-1023 .. s],  targets = y[s+1 .. s+30]
and y[s+30] = log(realized_var at origin s), so step 30 is the quantity
scored with QLIKE.
"""
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.data import target_30d
from src.metrics import qlike
from src.timestral_utils import quantile_mean

CONTEXT_LEN = 1024
HORIZON = 30
TRAIN_END = pd.Timestamp("2018-12-31")
VAL_START = pd.Timestamp("2019-01-01")
VAL_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class WindowSet:
    """A batch of (context, target) windows, possibly from several tickers.

    contexts:      (N, CONTEXT_LEN) float32, y up to and including the origin
    targets:       (N, HORIZON) float32, y over the next HORIZON days
    realized_var:  (N,) float64, mean daily variance over origin+1 .. origin+HORIZON
    tickers, origin_dates, target_end_dates: (N,) labels
    """

    contexts: np.ndarray
    targets: np.ndarray
    realized_var: np.ndarray
    tickers: np.ndarray
    origin_dates: pd.DatetimeIndex
    target_end_dates: pd.DatetimeIndex

    def __len__(self):
        return len(self.contexts)

    def subset(self, mask):
        return WindowSet(
            self.contexts[mask], self.targets[mask], self.realized_var[mask],
            self.tickers[mask], self.origin_dates[mask], self.target_end_dates[mask],
        )

    @staticmethod
    def concat(sets):
        return WindowSet(
            np.concatenate([s.contexts for s in sets]),
            np.concatenate([s.targets for s in sets]),
            np.concatenate([s.realized_var for s in sets]),
            np.concatenate([s.tickers for s in sets]),
            pd.DatetimeIndex(np.concatenate([s.origin_dates.values for s in sets])),
            pd.DatetimeIndex(np.concatenate([s.target_end_dates.values for s in sets])),
        )


def ticker_windows(daily_var, ticker, context_len=CONTEXT_LEN, horizon=HORIZON):
    """Every window of one ticker: one per origin s with a full context and full targets."""
    y = target_30d(daily_var, window=horizon)
    values = y.values
    n = len(values)
    origins = np.arange(context_len - 1, n - horizon)  # positions in y
    view = sliding_window_view(values, context_len + horizon)[origins - (context_len - 1)]

    # realized_var at origin s = mean(daily_var[s+1 .. s+horizon]), as src.data.realized_forward.
    realized = daily_var.rolling(horizon).mean().shift(-horizon).loc[y.index[origins]].values
    # y[s+horizon] is exactly log(realized_var at s): the identity notebook 02 relies on.
    np.testing.assert_allclose(view[:, -1], np.log(realized), rtol=1e-10)

    return WindowSet(
        contexts=view[:, :context_len].astype(np.float32),
        targets=view[:, context_len:].astype(np.float32),
        realized_var=realized.astype(np.float64),
        tickers=np.full(len(origins), ticker),
        origin_dates=y.index[origins],
        target_end_dates=y.index[origins + horizon],
    )


def split_windows(windows, val_step=5):
    """Train / validation windows; test-period targets are never included.

    train: every origin whose last target date is <= 2018-12-31.
    val:   origins in 2019-2021 whose last target date is <= 2021-12-31,
           every `val_step`-th origin per ticker (the spacing notebook 02
           uses for the test set).
    """
    train = windows.subset(windows.target_end_dates <= TRAIN_END)
    in_val = (windows.origin_dates >= VAL_START) & (windows.target_end_dates <= VAL_END)
    val_parts = []
    for ticker in pd.unique(windows.tickers):
        idx = np.flatnonzero(in_val & (windows.tickers == ticker))[::val_step]
        val_parts.append(windows.subset(idx))
    val = WindowSet.concat(val_parts)
    check_split_boundaries(train, val)
    return train, val


def check_split_boundaries(train, val):
    """Fail loudly if any target date crosses its split boundary."""
    assert len(train) > 0 and len(val) > 0
    assert train.target_end_dates.max() <= TRAIN_END, "a training target falls after 2018-12-31"
    assert val.origin_dates.min() >= VAL_START, "a validation origin falls before 2019-01-01"
    assert val.target_end_dates.max() <= VAL_END, "a validation target falls after 2021-12-31"
    assert val.target_end_dates.max() < TEST_START and train.target_end_dates.max() < TEST_START


def build_datasets(daily_vars, val_step=5):
    """`daily_vars`: {ticker: GK daily variance Series}. Returns (train, val) WindowSets."""
    all_windows = WindowSet.concat([ticker_windows(dv, t) for t, dv in daily_vars.items()])
    return split_windows(all_windows, val_step=val_step)


# ---------------------------------------------------------------------------
# Loss and evaluation (torch)
# ---------------------------------------------------------------------------


def pinball_loss(quantiles, targets, levels):
    """Mean pinball (quantile) loss over batch, horizon steps and quantile levels.

    quantiles: (B, 1, H, Q) or (B, H, Q) tensor, log space
    targets:   (B, H) tensor, log space
    levels:    the Q quantile levels (forecaster.quantile_levels)
    """
    import torch

    if quantiles.dim() == 4:
        quantiles = quantiles[:, 0]
    tau = torch.as_tensor(levels, dtype=quantiles.dtype, device=quantiles.device)
    diff = targets[..., None].to(quantiles.dtype) - quantiles  # (B, H, Q)
    return torch.maximum(tau * diff, (tau - 1) * diff).mean()


def predict(forecaster, contexts, horizon=HORIZON, batch_size=512):
    """(N, H, Q) numpy quantiles, fp32, no symmetric averaging, via the training code path."""
    import torch

    from src.timestral_utils import differentiable_quantiles

    out = []
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            x = torch.as_tensor(contexts[i : i + batch_size], device=forecaster.device)
            out.append(differentiable_quantiles(forecaster, x, horizon)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def point_forecasts(quantiles, levels, horizon=HORIZON):
    """Step-`horizon` forecasts in variance units, as in notebook 02.

    Returns dict with "mean" (quantile_mean of exp(quantiles)), "median",
    "q10", "q90".
    """
    levels = np.asarray(levels)
    values = np.exp(quantiles[:, horizon - 1, :])
    return {
        "mean": quantile_mean(values, levels),
        "median": values[:, int(np.argmin(np.abs(levels - 0.5)))],
        "q10": values[:, int(np.argmin(np.abs(levels - 0.10)))],
        "q90": values[:, int(np.argmin(np.abs(levels - 0.90)))],
    }


def evaluate(forecaster, windows, batch_size=512):
    """Validation pinball loss (log space, all steps) and QLIKE of the step-30 mean."""
    import torch

    levels = forecaster.quantile_levels
    q = predict(forecaster, windows.contexts, batch_size=batch_size)
    pinball = float(pinball_loss(torch.as_tensor(q), torch.as_tensor(windows.targets), levels))
    mean = point_forecasts(q, levels)["mean"]
    return {"pinball": pinball, "qlike": qlike(windows.realized_var, mean)}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _save(obj, path):
    import torch

    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous checkpoint


def train(
    forecaster,
    train_set,
    val_set,
    ckpt_dir,
    lr=1e-5,
    max_steps=3000,
    batch_size=32,
    eval_every=250,
    grad_clip=1.0,
    amp=True,
    patience=None,
    seed=0,
    log=print,
):
    """Full fine-tuning with pinball loss; keeps the checkpoint with the best val QLIKE.

    Writes to `ckpt_dir`:
      last.pt       model + optimizer + scaler + history, for resuming
      best.pt       model weights at the best validation QLIKE so far
      history.json  train / validation curves (readable without torch)
    If last.pt exists, training resumes from it. Batches are drawn with a
    generator seeded by (seed, step), so a resumed run sees the same
    batches as an uninterrupted one.

    `patience`: stop after this many evaluations without a new best val
    QLIKE (None = run the whole budget). Step 0 is evaluated before any
    update, so history[0] is the zero-shot (no symmetric averaging) model.
    """
    import torch

    from src.timestral_utils import differentiable_quantiles

    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    history_path = os.path.join(ckpt_dir, "history.json")

    model = forecaster.model
    device = forecaster.device
    levels = forecaster.quantile_levels
    use_amp = amp and device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    state = {"step": 0, "evals": [], "train": [], "best_qlike": float("inf"), "best_step": None,
             "skipped_steps": 0, "stopped_early": False, "config": {"lr": lr, "batch_size": batch_size, "grad_clip": grad_clip,
                                            "amp": use_amp, "seed": seed, "max_steps": max_steps}}
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        log(f"resumed from {last_path} at step {state['step']} (best val QLIKE {state['best_qlike']:.4f} "
            f"at step {state['best_step']})")

    contexts = torch.as_tensor(train_set.contexts)
    targets = torch.as_tensor(train_set.targets)
    n = len(contexts)

    def run_eval(step):
        metrics = evaluate(forecaster, val_set)
        recent = [loss for s, loss in state["train"] if s > step - eval_every]
        row = {"step": step, "val_pinball": metrics["pinball"], "val_qlike": metrics["qlike"],
               "train_pinball": float(np.mean(recent)) if recent and step > 0 else None}
        state["evals"].append(row)
        improved = metrics["qlike"] < state["best_qlike"]
        if improved:
            state["best_qlike"], state["best_step"] = metrics["qlike"], step
            _save({"model": model.state_dict(), "step": step, "val": metrics}, best_path)
        log(f"step {step:5d} | train pinball {row['train_pinball'] if row['train_pinball'] is not None else float('nan'):.5f}"
            f" | val pinball {metrics['pinball']:.5f} | val QLIKE {metrics['qlike']:.5f}" + (" *" if improved else ""))

    def checkpoint():
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "state": state}, last_path)
        with open(history_path, "w") as f:
            json.dump({k: v for k, v in state.items()}, f, indent=1, default=str)

    if state["step"] == 0 and not state["evals"]:
        run_eval(0)
        checkpoint()

    t0 = time.time()
    while state["step"] < max_steps and not state["stopped_early"]:
        step = state["step"] + 1
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        idx = torch.randint(n, (batch_size,), generator=gen)
        x, y = contexts[idx].to(device), targets[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                q = differentiable_quantiles(forecaster, x, HORIZON)
            loss = pinball_loss(q.float(), y, levels)
        except ValueError as e:  # _forecast_raw raises on NaN output (e.g. fp16 overflow)
            state["skipped_steps"] += 1
            log(f"step {step}: skipped ({e})")
            state["step"] = step
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        state["train"].append((step, float(loss.detach())))
        state["step"] = step

        if step % eval_every == 0 or step == max_steps:
            run_eval(step)
            checkpoint()
            log(f"        {(time.time() - t0) / 60:.1f} min elapsed this session")
            if patience is not None:
                evals_since_best = sum(1 for e in state["evals"] if e["step"] > state["best_step"])
                if evals_since_best >= patience:
                    log(f"early stop: no val QLIKE improvement in {patience} evaluations")
                    state["stopped_early"] = True
                    checkpoint()

    model.eval()
    return state
