"""
base.py — abstract Forecaster interface.

A Forecaster produces a PriceDistribution at a given horizon and spot price.
Subclasses implement different mechanisms (bootstrap, Merton jump-diffusion,
etc.) behind the same forecast() signature, so the downstream OptionValuer
is agnostic to which model produced the PDF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from options_trader.forecast.price_distribution import PriceDistribution


class Forecaster(ABC):
    """Abstract base for all forecasters.

    Concrete subclasses must implement forecast(), returning a PriceDistribution
    over the underlying's price at the given horizon.
    """

    @abstractmethod
    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        """Forecast the price distribution at `horizon_days` from `spot`.

        Args:
            horizon_days: integer number of trading days forward.
            spot: current price of the underlying (must be > 0).

        Returns:
            PriceDistribution over the underlying's price at the horizon.
        """
