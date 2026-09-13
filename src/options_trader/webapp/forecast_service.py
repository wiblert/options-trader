"""
forecast_service.py — build a JSON-serialisable forecast payload for the web UI.

PURE and network-free: every function here takes an already-loaded
`StockReturnTS` (the `server` module owns the `get_history` network call). That
keeps this layer unit-testable on synthetic data, in the same spirit as the
rest of the codebase.

What it produces, given a history + horizon:
  * the OHLC history bars to draw (a trailing window),
  * 30-ish bootstrap spaghetti price PATHS (mirrors `BootstrapForecaster`'s
    mechanism — iid weighted daily log-returns summed along the horizon),
  * per-step p10/p50/p90 path bands and a highlighted median path,
  * the TERMINAL price distribution as a smooth PDF + CDF on a common price
    grid, for both the empirical bootstrap and a Gaussian (normal/Black-Scholes)
    comparison built from the SAME ReturnDistribution,
  * in DIAGNOSTIC mode: the realized continuation (we hold out the last
    `horizon` bars), where the realized terminal lands as a PIT percentile in
    each forecast, and the KDE log score of the realized terminal — the project's
    own accuracy metrics, so the page actually diagnoses forecast quality.

The Gaussian terminal is exactly lognormal, so its PDF/CDF are analytic
(scipy.stats.lognorm); the bootstrap terminal PDF is a Gaussian KDE and its CDF
is the empirical CDF.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from options_trader.data.stock_return_ts import StockReturnTS


# Defaults tuned for snappy interactive use (not the 10k-path production size).
DEFAULT_N_PATHS = 2000
DEFAULT_N_SPAGHETTI = 30
DEFAULT_HISTORY_BARS = 120
DEFAULT_DECAY_LAMBDA = 0.99
GRID_POINTS = 240
CLIP_Q = (0.005, 0.995)


def _decay_weights(n: int, decay_lambda: float) -> np.ndarray:
    """Exponential-decay weights, newest (last) heaviest; normalised to sum 1.

    Mirrors ReturnDistribution's default weighting so the spaghetti/terminal
    distributions match what the production forecaster would draw from.
    """
    ages = np.arange(n - 1, -1, -1, dtype=float)  # oldest has largest age
    w = decay_lambda ** ages
    return w / w.sum()


def _weighted_moments(samples: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mu = float(np.sum(samples * weights))
    var = float(np.sum(weights * (samples - mu) ** 2))
    return mu, var


def _percentile_of(sorted_prices: np.ndarray, value: float) -> float:
    """Empirical CDF (fraction of mass at or below `value`) for uniform-weight samples."""
    return float(np.searchsorted(sorted_prices, value, side="right") / len(sorted_prices))


def _future_dates(anchor: pd.Timestamp, horizon: int,
                   realized_dates: Optional[np.ndarray]) -> list[str]:
    """ISO dates for the `horizon` forecast steps (excludes the anchor day).

    Diagnostic mode reuses the real realized dates so paths align with the
    realized line; live mode synthesises business days after the anchor.
    """
    if realized_dates is not None:
        return [pd.Timestamp(d).date().isoformat() for d in realized_dates]
    bdays = pd.bdate_range(anchor, periods=horizon + 1)[1:]
    return [d.date().isoformat() for d in bdays]


def build_forecast_payload(
    ts: StockReturnTS,
    *,
    horizon: int,
    mode: str = "diagnostic",
    n_paths: int = DEFAULT_N_PATHS,
    n_spaghetti: int = DEFAULT_N_SPAGHETTI,
    history_bars: int = DEFAULT_HISTORY_BARS,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    seed: int = 42,
) -> dict:
    """Build the full forecast payload for one ticker.

    Args:
        ts: loaded daily history (caller fetched it).
        horizon: trading days to forecast forward.
        mode: "diagnostic" holds out the last `horizon` bars as the realized
            continuation (so accuracy can be measured); "live" forecasts from
            the final bar with no realized overlay.
        n_paths: Monte-Carlo paths backing the terminal distribution.
        n_spaghetti: how many individual paths to draw as spaghetti lines.
        history_bars: trailing OHLC bars to include for context.
        decay_lambda: exponential-decay weighting of the return samples.
        seed: RNG seed (reproducible).

    Returns:
        A JSON-serialisable dict (see module docstring for the shape).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if mode not in ("diagnostic", "live"):
        raise ValueError(f"mode must be 'diagnostic' or 'live', got {mode!r}")

    close = np.asarray(ts.close, dtype=float)
    n = len(close)

    if mode == "diagnostic":
        # Hold out the last `horizon` bars as the realized future.
        cutoff = n - horizon - 1
        min_train = 30
        if cutoff < min_train:
            raise ValueError(
                f"not enough history for a {horizon}-day diagnostic on {ts.ticker} "
                f"({n} bars; need > {min_train + horizon})"
            )
    else:
        cutoff = n - 1

    train_close = close[: cutoff + 1]
    spot = float(train_close[-1])
    anchor_date = pd.Timestamp(ts.dates[cutoff])

    # Daily log-returns of the TRAINING window only (no look-ahead).
    log_returns = np.diff(np.log(train_close))
    weights = _decay_weights(len(log_returns), decay_lambda)

    rng = np.random.default_rng(seed)

    # --- Bootstrap paths: iid weighted daily log-returns, summed (additivity). ---
    draws = rng.choice(log_returns, size=(n_paths, horizon), replace=True, p=weights)
    log_paths = np.cumsum(draws, axis=1)                       # (n_paths, horizon)
    price_paths = spot * np.exp(log_paths)                     # terminal at [:, -1]
    # Prepend the anchor (step 0) so every path starts at spot.
    full_paths = np.concatenate(
        [np.full((n_paths, 1), spot), price_paths], axis=1     # (n_paths, horizon+1)
    )
    boot_terminal = full_paths[:, -1]

    # Per-step bands + median path (the highlighted central trajectory).
    p10 = np.percentile(full_paths, 10, axis=0)
    p50 = np.percentile(full_paths, 50, axis=0)
    p90 = np.percentile(full_paths, 90, axis=0)

    spaghetti = full_paths[: min(n_spaghetti, n_paths)].tolist()

    # --- Gaussian (normal / Black-Scholes) terminal: exactly lognormal. ---
    mu, var = _weighted_moments(log_returns, weights)
    sigma = float(np.sqrt(max(var, 1e-12)))
    k_mean = mu * horizon                       # mean of terminal LOG price - log(spot)
    k_std = sigma * np.sqrt(horizon)
    # lognorm params: shape=s=k_std, scale=spot*exp(k_mean)
    lognorm = stats.lognorm(s=k_std, scale=spot * np.exp(k_mean))

    # --- Common price grid for the PDF/CDF curves. ---
    union = np.concatenate([boot_terminal, lognorm.ppf([CLIP_Q[0], CLIP_Q[1]])])
    lo = float(np.quantile(boot_terminal, CLIP_Q[0]))
    hi = float(np.quantile(boot_terminal, CLIP_Q[1]))
    lo = min(lo, float(lognorm.ppf(CLIP_Q[0])))
    hi = max(hi, float(lognorm.ppf(CLIP_Q[1])))
    grid = np.linspace(lo, hi, GRID_POINTS)

    boot_kde = stats.gaussian_kde(boot_terminal)
    boot_pdf = boot_kde(grid)
    boot_sorted = np.sort(boot_terminal)
    boot_cdf = np.searchsorted(boot_sorted, grid, side="right") / len(boot_sorted)

    gauss_pdf = lognorm.pdf(grid)
    gauss_cdf = lognorm.cdf(grid)

    boot_median = float(np.median(boot_terminal))
    gauss_median = float(lognorm.median())

    realized_dates = (
        ts.dates[cutoff + 1: cutoff + 1 + horizon] if mode == "diagnostic" else None
    )

    payload: dict = {
        "ticker": ts.ticker,
        "mode": mode,
        "source": ts.source,
        "spot": spot,
        "anchor_date": anchor_date.date().isoformat(),
        "horizon": horizon,
        "n_paths": n_paths,
        "history": _history_bars(ts, cutoff, history_bars),
        "forecast_dates": _future_dates(anchor_date, horizon, realized_dates),
        "path_x": [anchor_date.date().isoformat()]
        + _future_dates(anchor_date, horizon, realized_dates),
        "spaghetti": spaghetti,
        "bands": {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()},
        "terminal": {
            "grid": grid.tolist(),
            "bootstrap": {
                "pdf": boot_pdf.tolist(),
                "cdf": boot_cdf.tolist(),
                "median": boot_median,
                "mean": float(boot_terminal.mean()),
                "q05": float(np.quantile(boot_terminal, 0.05)),
                "q95": float(np.quantile(boot_terminal, 0.95)),
            },
            "gaussian": {
                "pdf": gauss_pdf.tolist(),
                "cdf": gauss_cdf.tolist(),
                "median": gauss_median,
                "mean": float(lognorm.mean()),
                "q05": float(lognorm.ppf(0.05)),
                "q95": float(lognorm.ppf(0.95)),
            },
        },
        "diagnostics": None,
        "realized": None,
    }

    # --- Diagnostic overlay: realized continuation + accuracy metrics. ---
    if mode == "diagnostic":
        realized_close = close[cutoff: cutoff + horizon + 1]   # includes anchor
        realized_terminal = float(realized_close[-1])
        payload["realized"] = {
            "dates": [anchor_date.date().isoformat()]
            + [pd.Timestamp(d).date().isoformat() for d in realized_dates],
            "close": realized_close.tolist(),
            "terminal": realized_terminal,
        }
        boot_logscore = float(np.log(max(boot_kde(realized_terminal)[0], 1e-300)))
        gauss_logscore = float(lognorm.logpdf(realized_terminal))
        payload["diagnostics"] = {
            "realized_terminal": realized_terminal,
            "bootstrap": {
                "percentile": _percentile_of(boot_sorted, realized_terminal),
                "log_score": boot_logscore,
            },
            "gaussian": {
                "percentile": float(lognorm.cdf(realized_terminal)),
                "log_score": gauss_logscore,
            },
        }

    return payload


def _history_bars(ts: StockReturnTS, cutoff: int, history_bars: int) -> dict:
    """Trailing window of OHLC bars up to and including the anchor (index `cutoff`)."""
    start = max(0, cutoff - history_bars + 1)
    sl = slice(start, cutoff + 1)
    return {
        "dates": [pd.Timestamp(d).date().isoformat() for d in ts.dates[sl]],
        "open": np.asarray(ts.open[sl], dtype=float).tolist(),
        "high": np.asarray(ts.high[sl], dtype=float).tolist(),
        "low": np.asarray(ts.low[sl], dtype=float).tolist(),
        "close": np.asarray(ts.close[sl], dtype=float).tolist(),
    }
