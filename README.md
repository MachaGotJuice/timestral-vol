# timestral-vol

Can a time-series foundation model (Mistral's Timestral) forecast 30-day realized volatility as well as GARCH and HAR, and does a small fine-tune improve it?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MachaGotJuice/timestral-vol/blob/main/notebooks/00_setup_colab.ipynb)

## Motivation

TODO: why volatility forecasting, why a foundation model, why this matters for trading/risk.

## Data & target

TODO: tickers, Garman-Klass daily variance, the `target_30d` log-average-variance target, and the no-lookahead train/val/test split. See `src/data.py`.

## Models

- **GARCH(1,1)** -- TODO (`src/baselines.py`)
- **HAR-RV** -- TODO (`src/baselines.py`)
- **Timestral, zero-shot** -- TODO (`src/timestral_utils.py`, `notebooks/02_zero_shot.ipynb`)
- **Timestral, fine-tuned** -- TODO (`notebooks/03_finetuning.ipynb`)

## Evaluation

TODO: QLIKE loss, Diebold-Mariano significance testing, and the chronological time split used to avoid lookahead. See `src/metrics.py`.

## Results

TODO: tables/plots once baselines, zero-shot, and fine-tuned forecasts are in.

## Limitations

TODO.

## How to run

**Local (no GPU, no torch):** data, metrics, and tests only.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

**Colab (GPU, torch + Timestral):** open `notebooks/00_setup_colab.ipynb` via the badge above, set `Runtime > Change runtime type > T4 GPU`, and run all cells.
