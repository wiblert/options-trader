"""
option_implied_forecaster.py — the market's OWN risk-neutral PDF as a forecaster.

WHAT THIS IS
    Take a single ticker's own option chain, extract the Breeden-Litzenberger
    risk-neutral PDF of its terminal price at the option's expiry (see
    breeden_litzenberger.py), and re-anchor that PDF to the current spot. The result
    is a PriceDistribution that simply mirrors what the option market itself implies
    for S_T — no historical returns, no model, no Beta.

    r = log(K / pdf_spot),   S_T = spot · exp(r),   weight(r) = f_Q(K)·dK

WHY IT EXISTS — A BENCHMARK, NOT A PRODUCTION FORECASTER
    ⚠️ This must NEVER be wired as the production default. By construction it
    reproduces the very prices that are currently for sale, so valuing those same
    options against it is circular and yields ~zero edge by definition.

    Its sole purpose is a head-to-head accuracy check: run it on the rolling
    log-score backtest alongside our production forecaster (event-bootstrap). If our
    production forecast scores HIGHER than this option-implied forecast, then our view
    of S_T is more accurate than the market's own implied view — i.e. we have
    predictive edge over the prices for sale (and a basis to be profitable). If it
    does not, the market's price already knows what we know.

    (Contrast with option_implied_beta_forecaster.py, which maps the *index* options'
    PDF onto a ticker via Beta and adds idiosyncratic risk — that one is a genuine
    forecaster candidate. This one uses the ticker's OWN options directly, which is
    why it is only ever a benchmark.)

DETERMINISM
    The risk-neutral PDF is already the terminal distribution, so the forecast is a
    direct re-anchoring of the PDF grid — no Monte-Carlo sampling, no MC noise. The
    `n_paths`/`seed` arguments are accepted only to match the uniform forecaster
    constructor convention (and a future resample option); they do not affect output.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from options_trader.forecast.base import Forecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


logger = logging.getLogger(__name__)


DEFAULT_N_PATHS = 10_000


class OptionImpliedForecaster(Forecaster):
    """Forecaster that re-anchors an option-implied risk-neutral PDF to spot.

    Constructor:
        implied_returns: ReturnDistribution of horizon log-returns r = log(K/pdf_spot)
            weighted by the risk-neutral density — i.e. the output of
            `ImpliedPDF.to_return_distribution()` for the ticker's OWN chain.
        n_paths, seed: accepted for constructor uniformity with the other forecasters;
            the forecast is deterministic (the PDF grid IS the distribution), so these
            do not affect the result.
        pdf_horizon_days: the horizon the option PDF was derived for. When set and the
            requested forecast horizon differs, a warning is emitted — the PDF's width
            is fixed to its own expiry and does not rescale to another horizon.
    """

    def __init__(
        self,
        implied_returns: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
        *,
        pdf_horizon_days: Optional[int] = None,
    ) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        self.implied_returns = implied_returns
        self.n_paths = n_paths
        self.pdf_horizon_days = pdf_horizon_days
        self.rng = np.random.default_rng(seed)

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if self.pdf_horizon_days is not None and horizon_days != self.pdf_horizon_days:
            logger.warning(
                "forecast horizon (%d) != option-PDF horizon (%d); the option-implied "
                "distribution is fixed to its own expiry and will not rescale to the "
                "requested horizon",
                horizon_days, self.pdf_horizon_days,
            )

        # The PDF is already the terminal distribution at expiry: re-anchor the grid
        # to the current spot and carry the risk-neutral probabilities through as
        # PriceDistribution weights. No sampling — this is exact.
        terminal_prices = spot * np.exp(self.implied_returns.samples)

        logger.debug(
            "OptionImplied: horizon=%d spot=%.4f -> mean=%.4f std=%.4f (n=%d)",
            horizon_days, spot,
            float(np.sum(terminal_prices * self.implied_returns.weights)),
            float(np.sqrt(np.sum(
                self.implied_returns.weights
                * (terminal_prices - np.sum(terminal_prices * self.implied_returns.weights)) ** 2
            ))),
            len(terminal_prices),
        )
        return PriceDistribution(prices=terminal_prices, weights=self.implied_returns.weights)
