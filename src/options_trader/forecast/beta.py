"""
beta.py — historical Beta (β) of a ticker against benchmark indices.

Part of the Option-Implied Beta forecaster (see
`forecast/option_implied_beta_forecaster.py`). The market's forward-looking risk
PDF is extracted from highly-liquid INDEX options (SPY, IWM) and then mapped onto
an individual ticker by its Beta — how much the stock moves per unit of index
move:

    r_stock ≈ β · r_index + ε        (single-index / market model)

β is the slope of that regression, which equals

    β = Cov(r_stock, r_index) / Var(r_index)

over a historical window of daily log-returns. ε is the idiosyncratic residual
the index knows nothing about — the forecaster adds that back separately (the
"shrapnel"). This module computes only β; residuals live with the forecaster.

V1 methodology (deliberately simple, gated by the log-score backtest before it
becomes a default): a flat rolling window of daily log-returns, ordinary
covariance/variance. No shrinkage, no Vasicek adjustment, no Dimson lag for
non-synchronous trading — all are documented follow-ups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from options_trader.data.stock_return_ts import StockReturnTS


logger = logging.getLogger(__name__)


# ~1 trading year. Long enough for a stable slope, short enough to track a
# changing risk profile. Tunable (and ultimately backtest-gated).
DEFAULT_BETA_WINDOW = 252
MIN_BETA_OBS = 20  # below this a covariance slope is too noisy to trust


def log_returns(ts: StockReturnTS) -> tuple[np.ndarray, np.ndarray]:
    """Daily close-to-close log-returns of a StockReturnTS.

    Returns:
        (dates, returns) where dates are the dates OF each return (i.e. the
        later day of each close[i-1] -> close[i] pair, so dates align with the
        return that was realised that day) and returns is `diff(log(close))`.
    """
    close = np.asarray(ts.close, dtype=float)
    if len(close) < 2:
        raise ValueError(f"{ts.ticker}: need >= 2 closes for a return, got {len(close)}")
    if np.any(close <= 0):
        raise ValueError(f"{ts.ticker}: non-positive close price(s); cannot take log-returns")
    r = np.diff(np.log(close))
    return np.asarray(ts.dates[1:]), r


def align_log_returns(*series: StockReturnTS) -> tuple[np.ndarray, list[np.ndarray]]:
    """Inner-join several StockReturnTS on common return dates.

    Beta, covariance, and the idiosyncratic residual all require the stock and
    the index to be measured on the SAME days. Trading calendars can differ
    (holidays, data gaps), so we inner-join on date — the same discipline the
    VAR ReturnLoader used — rather than forward-fill (which would invent
    co-movement on a day one series didn't trade).

    Returns:
        (dates, [returns_0, returns_1, ...]) all aligned to the common dates,
        ordered oldest -> newest.
    """
    if len(series) < 1:
        raise ValueError("align_log_returns needs at least one series")

    cols = {}
    for i, ts in enumerate(series):
        d, r = log_returns(ts)
        cols[i] = pd.Series(r, index=pd.DatetimeIndex(d))
    df = pd.concat(cols, axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("no overlapping dates across the supplied series")

    dates = df.index.to_numpy()
    aligned = [df[i].to_numpy() for i in range(len(series))]
    return dates, aligned


def compute_beta(
    stock_returns: np.ndarray,
    index_returns: np.ndarray,
    window: Optional[int] = DEFAULT_BETA_WINDOW,
) -> float:
    """β = Cov(stock, index) / Var(index) over the last `window` observations.

    Args:
        stock_returns: 1-D aligned daily log-returns of the ticker.
        index_returns: 1-D aligned daily log-returns of the index (same length).
        window: use only the most recent `window` observations; None = all.

    Raises:
        ValueError: lengths differ, too few observations, or the index variance
            is degenerate (zero).
    """
    s = np.asarray(stock_returns, dtype=float)
    x = np.asarray(index_returns, dtype=float)
    if s.shape != x.shape or s.ndim != 1:
        raise ValueError(f"stock/index returns must be equal-length 1-D arrays, "
                         f"got {s.shape} and {x.shape}")
    if window is not None:
        s = s[-window:]
        x = x[-window:]
    if len(s) < MIN_BETA_OBS:
        raise ValueError(f"need >= {MIN_BETA_OBS} observations for beta, got {len(s)}")

    var_x = float(np.var(x, ddof=1))
    if var_x <= 0:
        raise ValueError("index return variance is zero; beta undefined")
    cov = float(np.cov(s, x, ddof=1)[0, 1])
    return cov / var_x


@dataclass(frozen=True)
class BetaResult:
    """Beta of the target ticker against each benchmark index."""

    beta_spy: float
    beta_iwm: float
    n_obs: int  # observations actually used (after windowing)

    def blended(self, weight_spy: float = 0.5) -> float:
        """Convex blend of the two betas (default 50/50, matching the 50/50 PDF blend)."""
        if not (0.0 <= weight_spy <= 1.0):
            raise ValueError(f"weight_spy must be in [0,1], got {weight_spy}")
        return weight_spy * self.beta_spy + (1.0 - weight_spy) * self.beta_iwm


class BetaCalculator:
    """Computes a ticker's Beta against SPY and IWM from aligned daily returns."""

    def __init__(self, window: Optional[int] = DEFAULT_BETA_WINDOW) -> None:
        self.window = window

    def compute(
        self,
        stock_ts: StockReturnTS,
        spy_ts: StockReturnTS,
        iwm_ts: StockReturnTS,
    ) -> BetaResult:
        """Beta of `stock_ts` vs SPY and IWM over the most recent `window` days."""
        _dates, (s, spy, iwm) = align_log_returns(stock_ts, spy_ts, iwm_ts)
        beta_spy = compute_beta(s, spy, self.window)
        beta_iwm = compute_beta(s, iwm, self.window)
        n = len(s) if self.window is None else min(len(s), self.window)
        logger.debug("Beta(%s): vs SPY=%.3f vs IWM=%.3f (n=%d)",
                     stock_ts.ticker, beta_spy, beta_iwm, n)
        return BetaResult(beta_spy=beta_spy, beta_iwm=beta_iwm, n_obs=n)
