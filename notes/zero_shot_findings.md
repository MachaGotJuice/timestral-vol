# Timestral zero-shot: findings and interpretation

Results from `notebooks/02_zero_shot.ipynb` run on Colab (10 tickers, 238
origins each, 2022-01-03 .. 2026-09-24, H=30, step=5), scored against the
same evaluation set as the GARCH/HAR baselines (`notebooks/01_data_and_baselines.ipynb`).

## Pooled QLIKE (lower is better)

| model | QLIKE |
|---|---|
| har | 0.1160 |
| ts209m_daily_mean | 0.1368 |
| ts5m_mean | 0.1422 |
| ts209m_mean | 0.1423 |
| ts209m_median | 0.1445 |
| garch (scaled) | 0.1507 |
| ts5m_median | 0.1514 |

HAR wins outright. Zero-shot Timestral roughly ties scaled GARCH and loses
to HAR.

## Per-ticker QLIKE

| ticker | garch | har | ts5m_median | ts5m_mean | ts209m_median | ts209m_mean | ts209m_daily_mean |
|---|---|---|---|---|---|---|---|
| SPY | 0.2183 | 0.1826 | 0.2649 | 0.2417 | 0.2409 | 0.2277 | 0.2105 |
| QQQ | 0.1631 | 0.1326 | 0.1794 | 0.1643 | 0.1730 | 0.1638 | 0.1503 |
| IWM | 0.1397 | 0.1168 | 0.1488 | 0.1480 | 0.1433 | 0.1493 | 0.1370 |
| EFA | 0.2132 | 0.1562 | 0.2205 | 0.1951 | 0.2097 | 0.1923 | 0.1860 |
| TLT | 0.0836 | 0.0720 | 0.0721 | 0.0701 | 0.0760 | 0.0790 | 0.0770 |
| GLD | 0.2353 | 0.1740 | 0.2391 | 0.2200 | 0.2284 | 0.2188 | 0.2133 |
| AAPL | 0.1317 | 0.1006 | 0.1177 | 0.1165 | 0.1143 | 0.1150 | 0.1096 |
| MSFT | 0.1240 | 0.0762 | 0.1042 | 0.0964 | 0.0990 | 0.0971 | 0.0949 |
| JPM | 0.0985 | 0.0886 | 0.0985 | 0.1004 | 0.0940 | 0.1008 | 0.0913 |
| XOM | 0.0997 | 0.0602 | 0.0691 | 0.0695 | 0.0663 | 0.0792 | 0.0985 |

## Diebold-Mariano: best Timestral variant (`ts209m_daily_mean`) vs. HAR and GARCH

| ticker | p vs HAR | p vs GARCH |
|---|---|---|
| SPY | 0.0592 | 0.6315 |
| QQQ | 0.0552 | 0.4044 |
| IWM | 0.1175 | 0.8295 |
| EFA | 0.1180 | 0.2782 |
| TLT | 0.5561 | 0.4877 |
| GLD | 0.0545 | 0.4427 |
| AAPL | 0.2724 | 0.2087 |
| MSFT | **0.0066** | **0.0281** |
| JPM | 0.7893 | 0.5132 |
| XOM | **0.0241** | 0.9506 |

**Pooled** (avg loss across tickers per origin, lag=5):
- vs HAR: **p = 0.0080** (significant -- HAR is genuinely better, not just numerically ahead)
- vs GARCH: p = 0.1216 (not significant -- roughly a tie)

## Calibration: fraction of `realized_var` inside the 10-90% band (target 80%)

| model | coverage |
|---|---|
| ts5m_mean | 0.778 |
| ts209m_mean | 0.786 |
| ts209m_daily_mean | 0.947 |

The smoothed variants are very well calibrated -- almost exactly on the
80% target. The daily-input variant is badly overcovered: its bands are
too wide (underconfident). QLIKE only scores the point forecast, so the
daily variant's lower QLIKE is a reasonably-centered point estimate
wrapped in bands too wide to trust.

## Leakage check

Timestral's GitHub repo (`mistralai/timestral`) has a single commit,
"Timestral release", dated **2026-10-15**. Our test period ends
2026-09-24 -- **0 of 11,900 observations fall after the release date**.
There is currently no post-release holdout window; pretraining-data
leakage on this exact financial history cannot be ruled out from the
public model card.

## Interpretation

**Does this mean Timestral was trained on financial data and can predict
volatility "for real," given the right input?**

Probably not, or at least the data doesn't support that reading. If
Timestral had meaningfully memorized this specific financial history, it
should crush HAR and GARCH, not tie one and lose to the other. The
smoothed-vs-daily gap is better explained as a generic time-series
property than financial domain knowledge: volatility is persistent /
mean-reverting, and *any* sequence model benefits from a less noisy,
trend-friendly input -- which is exactly why HAR's own weekly/monthly
trailing averages work, with no finance-specific knowledge involved.
Foundation models like this are typically trained on huge multi-domain
corpora (energy, retail, weather, etc.) where persistence shows up
everywhere; Timestral likely picked up "this kind of series is sticky,"
not "I've seen SPY before."

**Does this mean fine-tuning would add nothing?**

No -- if anything the opposite. The gap that matters is zero-shot
Timestral losing to HAR with pooled p = 0.008: a real, specific target
(30-day-average log variance) that a 4-coefficient linear regression
beats a transformer at. That's headroom, not a ceiling. The pattern most
worth fine-tuning is visible in the calibration numbers: the smoothed
variant's *uncertainty* is already excellent (78-79% vs. an 80% target)
but its *point forecast* is mediocre. A model with well-calibrated bands
but an off-center median/mean is the textbook case where even light
fine-tuning (or just an affine bias correction) tends to close a lot of
ground cheaply -- it doesn't need to fight the uncertainty estimates,
just recenter the point estimate.

**Bottom line:** zero-shot has clearly learned *something* generically
useful about volatility persistence (it's not random, it ties GARCH), but
there's a well-defined, specific gap left for fine-tuning to close. The
leakage risk cuts against the "Timestral is secretly great" theory, not
for it: even with a possible leakage advantage, zero-shot still doesn't
beat a simple linear regression.
