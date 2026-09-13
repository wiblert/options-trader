"""
return_distribution.py — weighted empirical distribution over daily log-returns.

The Monte Carlo forecaster draws from a ReturnDistribution to simulate K-day-ahead
price paths. Samples are stored as log-returns because log-returns are additive:
the sum of K iid draws is the K-day log-return.

Storage:
    samples: 1-D np.ndarray of daily log-returns (ordered oldest -> newest)
    weights: 1-D np.ndarray, parallel to samples, sums to 1
    _raw_weights: 1-D np.ndarray, unnormalized weights. Used internally so
        incremental updates (add_sample) preserve exact exponential-decay
        semantics.

Invariants (enforced on every mutation):
    * len(samples) == len(weights) == len(_raw_weights)
    * samples are finite
    * _raw_weights are non-negative and not all zero
    * weights = _raw_weights / _raw_weights.sum()
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


DEFAULT_DECAY_LAMBDA = 0.99
LARGE_RETURN_THRESHOLD = 0.5   # |log-return| > 0.5 (~65% one-day move) is flagged
JITTER_COPIES_DEFAULT = 10


class ReturnDistribution:
    """Weighted empirical distribution over daily log-returns."""

    samples: np.ndarray
    weights: np.ndarray
    _raw_weights: np.ndarray
    _decay_lambda: float
    label: Optional[str]

    def __init__(
        self,
        returns: np.ndarray,
        weights: Optional[np.ndarray] = None,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        label: Optional[str] = None,
    ) -> None:
        """Build a ReturnDistribution from a sequence of daily log-returns.

        Args:
            returns: 1-D array of log-returns, ordered oldest -> newest.
                Caller is responsible for converting prices -> log-returns;
                this class does not accept raw prices.
            weights: Optional 1-D array of weights. Stored as raw (unnormalised)
                weights and normalised for self.weights. If None, exponential-decay
                weights are computed from decay_lambda such that the most recent
                return carries the highest weight.
            decay_lambda: λ ∈ (0, 1). Weight at age a is proportional to λ^a,
                where age 0 is the most recent return. Used both to build default
                weights (when `weights` is None) and as the default decay factor
                in subsequent add_sample() calls. λ=0.99 implies ~69-day half-life.
            label: optional identifier (e.g. ticker symbol) included in warnings.
        """
        self.label = label
        self.samples = np.asarray(returns, dtype=float).copy()
        if self.samples.ndim != 1:
            raise ValueError(f"returns must be 1-D, got shape {self.samples.shape}")
        if len(self.samples) == 0:
            raise ValueError("returns must be non-empty")
        if not (0.0 < decay_lambda < 1.0):
            raise ValueError(f"decay_lambda must be in (0, 1), got {decay_lambda}")
        self._decay_lambda = decay_lambda

        n = len(self.samples)
        if weights is None:
            ages = np.arange(n - 1, -1, -1, dtype=float)  # oldest=n-1, newest=0
            self._raw_weights = decay_lambda ** ages
        else:
            raw = np.asarray(weights, dtype=float).copy()
            self._validate_raw_weights(raw)
            if len(raw) != n:
                raise ValueError(
                    f"weights length {len(raw)} != samples length {n}"
                )
            self._raw_weights = raw

        self.weights = self._raw_weights / self._raw_weights.sum()
        self.validate_samples()

    @staticmethod
    def _validate_raw_weights(w: np.ndarray) -> None:
        """Validate a raw weights array (non-negative, finite, sums to > 0)."""
        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights contain NaN/inf")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("weights sum to zero")

    # Kept for backward compatibility with anything that called the static helper.
    @classmethod
    def _normalize_weights(cls, w: np.ndarray) -> np.ndarray:
        cls._validate_raw_weights(w)
        return w / w.sum()

    @classmethod
    def from_normal_distribution(
        cls,
        mean: float,
        variance: float,
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ) -> "ReturnDistribution":
        """Draw `n_samples` from N(mean, variance) with uniform weights.

        Useful as a Black-Scholes-equivalent baseline and for testing.
        """
        if variance < 0:
            raise ValueError(f"variance must be non-negative, got {variance}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        rng = np.random.default_rng(seed)
        samples = rng.normal(loc=mean, scale=np.sqrt(variance), size=n_samples)
        weights = np.ones(n_samples)  # uniform raw weights
        return cls(returns=samples, weights=weights)

    def uniform_weights(self) -> None:
        """Reset weights to uniform 1/N. Resets raw weights to all-ones."""
        n = len(self.samples)
        self._raw_weights = np.ones(n)
        self.weights = self._raw_weights / n

    def add_sample(
        self,
        new_return: float,
        decay_lambda: Optional[float] = None,
    ) -> None:
        """Append a new (newest) log-return and exponentially decay existing weights.

        Mathematically equivalent to rebuilding the distribution from scratch with
        the extended sample series — confirmed by the
        `test_add_sample_matches_rebuild_from_scratch` test.

        Args:
            new_return: the new daily log-return (age 0 after this call).
            decay_lambda: optional override for the decay factor. If None, uses
                the decay_lambda stored on the instance at construction.
        """
        if not np.isfinite(new_return):
            raise ValueError(f"new_return must be finite, got {new_return}")

        if decay_lambda is None:
            decay_lambda = self._decay_lambda
        if not (0.0 < decay_lambda < 1.0):
            raise ValueError(f"decay_lambda must be in (0, 1), got {decay_lambda}")

        # Mathematically: decay existing raw weights by λ (their ages all increased
        # by 1), then append the new sample with raw weight 1 (λ^0). The result is
        # exactly λ^age for the extended series.
        self._raw_weights = np.concatenate(
            [self._raw_weights * decay_lambda, np.array([1.0])]
        )
        self.samples = np.append(self.samples, float(new_return))
        self.weights = self._raw_weights / self._raw_weights.sum()

        self.validate_samples()

    def validate_samples(self) -> None:
        """Check that samples are within reasonable bounds.

        Raises:
            ValueError: any sample is NaN or inf.
        Logs warning when |log-return| > LARGE_RETURN_THRESHOLD.
        """
        if not np.all(np.isfinite(self.samples)):
            n_bad = int((~np.isfinite(self.samples)).sum())
            raise ValueError(f"samples contain {n_bad} NaN/inf value(s)")

        n_large = int((np.abs(self.samples) > LARGE_RETURN_THRESHOLD).sum())
        if n_large > 0:
            prefix = f"{self.label}: " if self.label else ""
            logger.warning(
                "%s%d sample(s) have |log-return| > %g (~65%% one-day move); "
                "consider data-quality check",
                prefix, n_large, LARGE_RETURN_THRESHOLD,
            )

    def smooth_samples(
        self,
        bandwidth: Optional[float] = None,
        jitter_copies: int = JITTER_COPIES_DEFAULT,
        seed: Optional[int] = None,
    ) -> None:
        """Apply Gaussian smoothing by appending jittered copies of each sample.

        For each existing sample x_i with raw weight r_i, draw `jitter_copies`
        new samples from N(x_i, bandwidth^2) and append them with raw weight r_i
        each. All weights are then renormalised so the total is 1 — this is
        equivalent to spreading each original r_i across (1 + jitter_copies)
        copies.

        Caveat: after smoothing, subsequent add_sample() calls still update raw
        weights correctly, but the appended sample is unsmoothed (a single point),
        which may not match the smoothed structure of existing samples. Best
        practice for rolling backtests: do all add_sample() updates first, then
        smooth a copy at forecast time.
        """
        if bandwidth is None:
            bandwidth = self._silverman_bandwidth()
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
        if jitter_copies < 1:
            raise ValueError(f"jitter_copies must be >= 1, got {jitter_copies}")

        rng = np.random.default_rng(seed)
        n = len(self.samples)

        jitter = rng.normal(loc=0.0, scale=bandwidth, size=(n, jitter_copies))
        jittered = self.samples[:, None] + jitter
        new_raw = np.repeat(self._raw_weights[:, None], jitter_copies, axis=1)

        self.samples = np.concatenate([self.samples, jittered.ravel()])
        self._raw_weights = np.concatenate([self._raw_weights, new_raw.ravel()])
        self.weights = self._raw_weights / self._raw_weights.sum()

    def _silverman_bandwidth(self) -> float:
        """Silverman's rule of thumb: h = 1.06 · σ̂ · n^(-1/5).

        σ̂ is the weighted sample standard deviation.
        """
        n = len(self.samples)
        mean = float(np.sum(self.samples * self.weights))
        var = float(np.sum(self.weights * (self.samples - mean) ** 2))
        sigma = np.sqrt(var)
        if sigma <= 0:
            raise ValueError("cannot compute Silverman bandwidth: weighted variance is 0")
        return 1.06 * sigma * n ** (-1.0 / 5.0)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        mean = float(np.sum(self.samples * self.weights))
        var = float(np.sum(self.weights * (self.samples - mean) ** 2))
        ess = 1.0 / float((self.weights ** 2).sum())   # Kish effective sample size
        return (
            f"ReturnDistribution(n={len(self)}, ess={ess:.1f}, "
            f"mean={mean:.5f}, std={np.sqrt(var):.5f})"
        )
