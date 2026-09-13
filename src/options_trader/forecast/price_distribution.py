"""
price_distribution.py — weighted empirical distribution over terminal prices.

Produced by Forecasters (the output type of every Forecaster.forecast() call).
Consumed by the OptionValuer to compute expected payoffs and tail risk.

Mirrors the ReturnDistribution shape but over prices instead of log-returns,
so prices must be strictly positive.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PriceDistribution:
    """Weighted empirical distribution over terminal prices."""

    prices: np.ndarray
    weights: np.ndarray

    def __init__(
        self,
        prices: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """Build a PriceDistribution from a sequence of terminal prices.

        Args:
            prices: 1-D array of strictly positive terminal prices.
            weights: Optional 1-D weights, normalised to sum to 1.
                Defaults to uniform 1/N.
        """
        self.prices = np.asarray(prices, dtype=float).copy()
        if self.prices.ndim != 1:
            raise ValueError(f"prices must be 1-D, got shape {self.prices.shape}")
        if len(self.prices) == 0:
            raise ValueError("prices must be non-empty")
        if not np.all(np.isfinite(self.prices)):
            n_bad = int((~np.isfinite(self.prices)).sum())
            raise ValueError(f"prices contain {n_bad} NaN/inf value(s)")
        if np.any(self.prices <= 0):
            n_bad = int((self.prices <= 0).sum())
            raise ValueError(f"prices must be positive, got {n_bad} non-positive value(s)")

        n = len(self.prices)
        if weights is None:
            self.weights = np.ones(n) / n
        else:
            self.weights = self._normalize_weights(np.asarray(weights, dtype=float))
            if len(self.weights) != n:
                raise ValueError(
                    f"weights length {len(self.weights)} != prices length {n}"
                )

    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights contain NaN/inf")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero")
        return w / total

    def mean(self) -> float:
        return float(np.sum(self.prices * self.weights))

    def std(self) -> float:
        m = self.mean()
        return float(np.sqrt(np.sum(self.weights * (self.prices - m) ** 2)))

    def quantile(self, q: float) -> float:
        """Weighted quantile (0 <= q <= 1)."""
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0,1], got {q}")
        order = np.argsort(self.prices)
        return float(np.interp(q, np.cumsum(self.weights[order]), self.prices[order]))

    def __len__(self) -> int:
        return len(self.prices)

    def __repr__(self) -> str:
        return (
            f"PriceDistribution(n={len(self)}, mean={self.mean():.2f}, "
            f"std={self.std():.2f}, 5%={self.quantile(0.05):.2f}, "
            f"95%={self.quantile(0.95):.2f})"
        )
