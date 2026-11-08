"""Section-1 checks for differentiable_quantiles (torch + timestral + Timestral-5M weights).

Run on Colab:  pytest -m torch tests/test_differentiable.py
Skipped automatically in the torch-free local environment.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timestral")

from src.finetune import pinball_loss  # noqa: E402
from src.timestral_utils import differentiable_quantiles, load_model  # noqa: E402

pytestmark = pytest.mark.torch

T, H = 1024, 30


def _contexts(batch, seed=0):
    """Log-variance-like series: a slow AR(1) around log(1e-4) plus noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros((batch, T + H))
    for t in range(1, T + H):
        x[:, t] = 0.99 * x[:, t - 1] + 0.1 * rng.standard_normal(batch)
    return (np.log(1e-4) + x).astype(np.float32)


@pytest.fixture
def forecaster():
    torch.manual_seed(0)
    return load_model("5M")


def test_matches_forecast_without_symmetric_averaging(forecaster):
    ctx = _contexts(8)[:, :T]
    ctx[0, :50] = np.nan  # missing values must be handled the same way
    expected = forecaster.forecast(ctx, prediction_length=H, symmetric_averaging=False,
                                   clip_nonnegative=False).quantiles
    got = differentiable_quantiles(forecaster, torch.as_tensor(ctx, device=forecaster.device), H)
    assert got.shape == (8, 1, H, len(forecaster.quantile_levels))
    assert got.requires_grad
    torch.testing.assert_close(got.detach(), expected, atol=1e-5, rtol=1e-5)


def test_gradients_reach_model_parameters(forecaster):
    data = torch.as_tensor(_contexts(4), device=forecaster.device)
    q = differentiable_quantiles(forecaster, data[:, :T], H)
    pinball_loss(q, data[:, T:], forecaster.quantile_levels).backward()

    grads = [p.grad for p in forecaster.model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    nonzero = [g.abs().sum().item() > 0 for g in grads]
    assert sum(nonzero) / len(nonzero) > 0.9, f"only {sum(nonzero)}/{len(nonzero)} tensors got gradient"


def test_overfits_one_small_batch(forecaster):
    data = torch.as_tensor(_contexts(16, seed=1), device=forecaster.device)
    x, y = data[:, :T], data[:, T:]
    opt = torch.optim.AdamW(forecaster.model.parameters(), lr=1e-4, weight_decay=0.0)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = pinball_loss(differentiable_quantiles(forecaster, x, H), y, forecaster.quantile_levels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forecaster.model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    print(f"pinball: step 0 {losses[0]:.4f} -> step 200 {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
