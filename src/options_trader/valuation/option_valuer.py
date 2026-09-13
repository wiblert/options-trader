"""
option_valuer.py — turn a forecast PriceDistribution into an option EV + edge.

This is the differentiator of the project. The pipeline is:

    history → forecaster → PriceDistribution (P-measure, at the option's expiry)
            → OptionValuer → OptionValuation (EV, edge vs market) → Kelly sizer

KEY DISTINCTION: the `fair_value` here is the expected value of the option's
terminal payoff UNDER OUR FORECAST (the real-world / P measure), discounted to
today. It is NOT a no-arbitrage price (that would be the risk-neutral / Q
expectation, which is what bs_price and the market quote give). The edge is the
gap between the two: we buy when our P-measure EV exceeds what the market charges.

V1 assumptions (see docs/decisions.md):
  * Held-to-expiry: the PriceDistribution must be forecast at the horizon equal
    to this contract's expiry. The terminal payoff valuation is then exact and
    needs no price path. The caller owns that consistency.
  * European exercise. For calls on non-dividend payers early exercise is never
    optimal, so this is exact there; with dividends it is a small approximation.
  * Flat risk-free rate for discounting.
  * Per-share valuation, matching how option premiums (bid/ask) are quoted.
    The 100x contract multiplier is a dollar-sizing concern for the Kelly layer,
    not an edge concern, so it lives there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np

from options_trader.forecast.price_distribution import PriceDistribution


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Recommendation(str, Enum):
    BUY = "BUY"        # our EV exceeds the ask — worth paying up for
    SELL = "SELL"      # our EV is below the bid — overpriced, sell/avoid
    HOLD = "HOLD"      # EV sits inside the spread — no edge after costs
    NO_QUOTE = "NO_QUOTE"  # no tradable bid/ask available


@dataclass(frozen=True)
class OptionContract:
    """Static contract terms plus (optionally) the current market quote.

    The quote fields are populated by the options-chain data layer. They are
    Optional so a contract can be valued for EV alone before a live quote exists.
    """

    symbol: str
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    bid: Optional[float] = None
    ask: Optional[float] = None
    implied_vol: Optional[float] = None
    open_interest: Optional[int] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return 0.5 * (self.bid + self.ask)


@dataclass(frozen=True)
class OptionValuation:
    """Result of valuing one contract against one PriceDistribution."""

    contract: OptionContract
    spot: float
    days_to_expiry: int
    expected_payoff: float            # undiscounted E_P[max(S_T - K, 0)] (per share)
    fair_value: float                 # discounted EV under our forecast (per share)
    prob_itm: float                   # P(option expires in the money)

    # market-relative fields (None when no quote)
    buy_price: Optional[float]        # the ask — what we pay to open a long
    sell_price: Optional[float]       # the bid — what we receive to open a short
    edge_buy: Optional[float]         # fair_value - ask  (>0 ⇒ underpriced)
    edge_sell: Optional[float]        # bid - fair_value  (>0 ⇒ overpriced)
    edge_pct_buy: Optional[float]     # edge_buy / ask == fair_value / ask - 1 (return on premium paid)
    breakeven: Optional[float]        # underlying price at expiry to break even on a long
    recommendation: Recommendation

    def __repr__(self) -> str:
        edge = f"{self.edge_buy:+.3f}" if self.edge_buy is not None else "n/a"
        return (
            f"OptionValuation({self.contract.symbol} {self.recommendation.value} "
            f"fair={self.fair_value:.3f} ask={self.buy_price} edge_buy={edge} "
            f"p_itm={self.prob_itm:.2%})"
        )


@dataclass(frozen=True)
class EdgeDecomposition:
    """Splits an option's edge into a DRIFT part and a SHAPE part.

    Our forecast distribution can differ from the market's in two ways: where it
    is centered (drift — our mean vs the risk-neutral forward) and its dispersion/
    skew/tails (shape). Re-centering our distribution onto the forward (a pure
    multiplicative rescale that preserves shape) and re-valuing isolates the two:

        total_edge = drift_edge + shape_edge

    Why it matters: the drift part is the noisy historical-trend bet we distrust
    (docs/issues.md ISSUE-1); the shape part is a realized-vs-implied-vol / tail
    disagreement. Reporting the split per signal shows how much of an edge is the
    part we have reason to doubt.
    """

    contract: OptionContract
    forward: float                       # risk-neutral forward = spot·e^{rT}
    forecast_mean: float                 # mean of our (as-is) terminal distribution
    fair_value: float                    # EV under our distribution
    fair_value_recentered: float         # EV after centering our dist on the forward
    drift_edge: float                    # fair_value - fair_value_recentered
    buy_price: Optional[float]           # the ask (None if no quote)
    total_edge: Optional[float]          # fair_value - ask
    shape_edge: Optional[float]          # fair_value_recentered - ask
    drift_share: Optional[float]         # drift_edge / total_edge (None if total≈0)

    def __repr__(self) -> str:
        ds = f"{self.drift_share:.0%}" if self.drift_share is not None else "n/a"
        return (
            f"EdgeDecomposition({self.contract.symbol} total={self.total_edge} "
            f"drift={self.drift_edge:+.2f} shape={self.shape_edge} drift_share={ds})"
        )


class OptionValuer:
    """Computes EV and market edge for option contracts from a PriceDistribution."""

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Args:
            risk_free_rate: continuously-compounded annual rate used to discount
                the expected terminal payoff to present value.
        """
        self.risk_free_rate = float(risk_free_rate)

    def value(
        self,
        price_dist: PriceDistribution,
        contract: OptionContract,
        spot: float,
        valuation_date: date,
    ) -> OptionValuation:
        """Value `contract` against `price_dist`.

        Args:
            price_dist: terminal-price distribution forecast at this contract's
                expiry (the held-to-expiry invariant — the caller's responsibility).
            contract: the option to value, optionally carrying a market quote.
            spot: current underlying price (used for breakeven/diagnostics only).
            valuation_date: today; T = (expiry - valuation_date) / 365 for discounting.

        Returns:
            OptionValuation. Market-relative fields are None when no quote is set.
        """
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")

        days_to_expiry = (contract.expiry - valuation_date).days
        if days_to_expiry < 0:
            raise ValueError(
                f"contract expired: expiry {contract.expiry} before valuation date {valuation_date}"
            )
        t_years = days_to_expiry / 365.0

        prices = price_dist.prices
        weights = price_dist.weights
        strike = contract.strike

        if contract.option_type == OptionType.CALL:
            payoffs = np.maximum(prices - strike, 0.0)
            itm = prices > strike
        else:
            payoffs = np.maximum(strike - prices, 0.0)
            itm = prices < strike

        expected_payoff = float(np.sum(weights * payoffs))
        prob_itm = float(np.sum(weights[itm]))
        discount = np.exp(-self.risk_free_rate * t_years)
        fair_value = float(discount * expected_payoff)

        ask = contract.ask
        bid = contract.bid
        buy_price = ask
        sell_price = bid

        edge_buy = edge_sell = edge_pct_buy = breakeven = None
        recommendation = Recommendation.NO_QUOTE

        if ask is not None and ask > 0:
            edge_buy = fair_value - ask
            edge_pct_buy = edge_buy / ask
            breakeven = (
                strike + ask if contract.option_type == OptionType.CALL else strike - ask
            )
        if bid is not None and bid > 0:
            edge_sell = bid - fair_value

        # Decision: only positive net-of-spread edges count. Buying clears the
        # ask, selling must beat the bid; an EV inside the spread is no edge.
        if edge_buy is not None and edge_buy > 0:
            recommendation = Recommendation.BUY
        elif edge_sell is not None and edge_sell > 0:
            recommendation = Recommendation.SELL
        elif ask is not None or bid is not None:
            recommendation = Recommendation.HOLD

        return OptionValuation(
            contract=contract,
            spot=spot,
            days_to_expiry=days_to_expiry,
            expected_payoff=expected_payoff,
            fair_value=fair_value,
            prob_itm=prob_itm,
            buy_price=buy_price,
            sell_price=sell_price,
            edge_buy=edge_buy,
            edge_sell=edge_sell,
            edge_pct_buy=edge_pct_buy,
            breakeven=breakeven,
            recommendation=recommendation,
        )

    def forward_price(self, spot: float, valuation_date: date, expiry: date) -> float:
        """Risk-neutral forward = spot·e^{rT}, T = (expiry - valuation_date)/365."""
        t_years = (expiry - valuation_date).days / 365.0
        return float(spot * np.exp(self.risk_free_rate * t_years))

    def decompose_edge(
        self,
        price_dist: PriceDistribution,
        contract: OptionContract,
        spot: float,
        valuation_date: date,
    ) -> EdgeDecomposition:
        """Split this contract's edge into drift vs shape (see EdgeDecomposition).

        Re-centers our distribution onto the forward by the multiplicative factor
        forward / E[S_T] — preserving relative shape (skew, tails, coefficient of
        variation) while removing the directional drift — and re-values.
        """
        base = self.value(price_dist, contract, spot, valuation_date)
        forward = self.forward_price(spot, valuation_date, contract.expiry)

        factor = forward / price_dist.mean()
        recentered = PriceDistribution(price_dist.prices * factor, price_dist.weights)
        reset = self.value(recentered, contract, spot, valuation_date)

        drift_edge = base.fair_value - reset.fair_value
        buy_price = base.buy_price
        total_edge = shape_edge = drift_share = None
        if buy_price is not None and buy_price > 0:
            total_edge = base.fair_value - buy_price
            shape_edge = reset.fair_value - buy_price
            # share of the total edge attributable to drift; undefined when the
            # total edge is ~0 (drift and shape can offset).
            drift_share = drift_edge / total_edge if abs(total_edge) > 1e-9 else None

        return EdgeDecomposition(
            contract=contract,
            forward=forward,
            forecast_mean=float(price_dist.mean()),
            fair_value=base.fair_value,
            fair_value_recentered=reset.fair_value,
            drift_edge=drift_edge,
            buy_price=buy_price,
            total_edge=total_edge,
            shape_edge=shape_edge,
            drift_share=drift_share,
        )
