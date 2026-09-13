"""
option_implied_beta_forecaster.py — map the market's option-implied risk onto a
single ticker via Beta, then add back its idiosyncratic risk.

THE IDEA (see breeden_litzenberger.py and beta.py for the pieces)
    Individual mid-cap option chains are illiquid — a directly-extracted PDF is
    noise. Index options (SPY large-cap, IWM small-cap) are the most liquid
    derivatives in the world and carry a smooth, forward-looking PDF of market
    volatility, skew, and fat tails. We:

      1. BLEND. Take the Breeden-Litzenberger risk-neutral PDFs of SPY and IWM and
         combine them 50/50 (V1) into one market-return distribution that spans both
         large- and small-cap systematic risk.
      2. DRAW & SCALE. Per Monte-Carlo path, draw a market return r_m from the blend
         and scale it by the ticker's Beta:  β · r_m  is the systematic component.
      3. ADD IDIOSYNCRATIC "SHRAPNEL". The index knows nothing about single-name risk
         (earnings jumps, news). Add a draw from the ticker's own historical residual
         ε = r_stock − β · r_market — the part the market model leaves unexplained.

         r_stock = β · r_market + ε,   S_T = spot · exp(r_stock)

HORIZON CONSISTENCY (important)
    The index PDF is for the option's expiry, so its draw is already the WHOLE-horizon
    market return — one draw per path, NOT summed. The idiosyncratic residuals are
    DAILY, so we sum `horizon_days` iid residual draws (log-return additivity, like
    the bootstrap). The forecaster is therefore built for ONE horizon: pass the same
    `horizon_days` to `forecast()` that the index PDF was derived for. A mismatch is
    logged (the systematic width would no longer match the requested horizon).

MEASURE NOTE
    The systematic component is risk-neutral (Q) — it carries a variance risk
    premium vs the realised (P) market move. V1 uses it as-is (the spec's design);
    de-meaning the index draw onto the realised forward, or subtracting a calibrated
    VRP, is a documented V2 refinement to validate on the log-score backtest.

STATUS: NOT a production default. Per the standing decision criterion it must first
clear the rolling out-of-sample log-score backtest vs the incumbent — which needs
historical index option chains (data/options_history.py) to derive the PDF on each
past run date. Until then this is a built, tested, ready-to-evaluate forecaster.
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


class OptionImpliedBetaForecaster(Forecaster):
    """Monte-Carlo forecaster: blended index PDF · Beta + idiosyncratic residual.

    Constructor:
        spy_market: ReturnDistribution of SPY horizon log-returns (from
            ImpliedPDF.to_return_distribution()).
        iwm_market: ReturnDistribution of IWM horizon log-returns.
        beta_spy, beta_iwm: the ticker's Beta vs each index.
        idiosyncratic: ReturnDistribution of DAILY residual log-returns
            ε = r_stock − β_blend · r_market_blend.
        n_paths, seed: Monte-Carlo controls.
        blend_spy: weight on SPY in the 50/50 (default) PDF + Beta blend.
        pdf_horizon_days: the horizon the index PDF was derived for (for the
            consistency check in forecast()); None disables the check.
        idio_event_dists: optional {EventType: ReturnDistribution} of event-day
            IDIOSYNCRATIC residuals (e.g. the ticker's historical earnings-day
            residuals). Only single-name events belong here — MARKET events (FOMC,
            CPI) are already priced into the index PDF on the systematic side, so
            conditioning the idiosyncratic side on them would double-count.
        event_schedule: per-horizon-day list[EventType|None] (length horizon_days);
            a day routes its idiosyncratic draw to idio_event_dists[type] when set.
            Supply both idio_event_dists and event_schedule to enable event mode.
    """

    def __init__(
        self,
        spy_market: ReturnDistribution,
        iwm_market: ReturnDistribution,
        beta_spy: float,
        beta_iwm: float,
        idiosyncratic: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
        *,
        blend_spy: float = 0.5,
        pdf_horizon_days: Optional[int] = None,
        idio_event_dists: Optional[dict] = None,
        event_schedule: Optional[list] = None,
    ) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        if not (0.0 <= blend_spy <= 1.0):
            raise ValueError(f"blend_spy must be in [0,1], got {blend_spy}")

        self.beta_spy = float(beta_spy)
        self.beta_iwm = float(beta_iwm)
        self.blend_spy = float(blend_spy)
        self.blended_beta = blend_spy * self.beta_spy + (1.0 - blend_spy) * self.beta_iwm
        self.idiosyncratic = idiosyncratic
        self.idio_event_dists = idio_event_dists or {}
        self.event_schedule = list(event_schedule) if event_schedule is not None else None
        self.n_paths = n_paths
        self.pdf_horizon_days = pdf_horizon_days
        self.rng = np.random.default_rng(seed)

        # Pre-build the 50/50 blended market-return pool: concatenate both index
        # sample pools, scaling each pool's weights so SPY carries blend_spy of the
        # total mass and IWM the rest. Drawing iid from this pool == flipping a
        # blend_spy-weighted coin to pick the index, then drawing from it.
        w_spy = spy_market.weights * blend_spy
        w_iwm = iwm_market.weights * (1.0 - blend_spy)
        self._market_samples = np.concatenate([spy_market.samples, iwm_market.samples])
        self._market_weights = np.concatenate([w_spy, w_iwm])
        self._market_weights = self._market_weights / self._market_weights.sum()

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if self.pdf_horizon_days is not None and horizon_days != self.pdf_horizon_days:
            logger.warning(
                "forecast horizon (%d) != index-PDF horizon (%d); the systematic "
                "component is a whole-horizon draw for the PDF's expiry and will not "
                "rescale to the requested horizon",
                horizon_days, self.pdf_horizon_days,
            )
        if self.event_schedule is not None and len(self.event_schedule) != horizon_days:
            raise ValueError(
                f"event_schedule length {len(self.event_schedule)} != horizon_days {horizon_days}"
            )

        # 1+2. Systematic: one whole-horizon market draw per path, scaled by Beta.
        # (Market events — FOMC/CPI — are already inside the index PDF here.)
        market = self.rng.choice(
            self._market_samples, size=self.n_paths, replace=True, p=self._market_weights
        )
        systematic = self.blended_beta * market

        # 3. Idiosyncratic: sum of horizon_days iid daily residual draws (additivity).
        if self.event_schedule is None:
            idio_daily = self.rng.choice(
                self.idiosyncratic.samples, size=(self.n_paths, horizon_days),
                replace=True, p=self.idiosyncratic.weights,
            )
            idiosyncratic = idio_daily.sum(axis=1)
        else:
            # Event mode: route each horizon day's idiosyncratic draw to the matching
            # single-name event distribution (e.g. earnings), else the normal residual.
            idiosyncratic = np.zeros(self.n_paths)
            for k in range(horizon_days):
                etype = self.event_schedule[k]
                rd = self.idio_event_dists.get(etype, self.idiosyncratic) if etype is not None \
                    else self.idiosyncratic
                idiosyncratic += self.rng.choice(rd.samples, size=self.n_paths,
                                                 replace=True, p=rd.weights)

        total_log_return = systematic + idiosyncratic
        terminal_prices = spot * np.exp(total_log_return)

        logger.debug(
            "OptionImpliedBeta: β_spy=%.3f β_iwm=%.3f β_blend=%.3f horizon=%d spot=%.4f "
            "-> mean=%.4f std=%.4f",
            self.beta_spy, self.beta_iwm, self.blended_beta, horizon_days, spot,
            float(terminal_prices.mean()), float(terminal_prices.std()),
        )
        return PriceDistribution(prices=terminal_prices)


def idiosyncratic_residuals(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    blended_beta: float,
) -> np.ndarray:
    """ε = r_stock − β · r_market over aligned daily log-returns (the "shrapnel").

    The leftover single-name risk after removing the market's daily impact — drawn
    from empirically (fat tails, earnings jumps preserved) by the forecaster.
    """
    s = np.asarray(stock_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    if s.shape != m.shape or s.ndim != 1:
        raise ValueError(f"stock/market returns must be equal-length 1-D, got {s.shape}, {m.shape}")
    return s - blended_beta * m
