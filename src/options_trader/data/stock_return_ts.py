"""
stock_return_ts.py — passive container for a single ticker's daily OHLCV time series.

Holds the raw historical data pulled by get_history. Downstream modules
(ReturnDistribution, forecasters, valuers) consume instances of this class.
No computation lives here — it is a typed bag of numpy arrays.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


# Array-valued fields on StockReturnTS that must align with `dates`.
# Update here when adding or removing aligned arrays on the class.
OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass
class StockReturnTS:
    ticker: str
    dates: np.ndarray       # datetime64[ns], one per trading day
    open: np.ndarray        # float64
    high: np.ndarray        # float64
    low: np.ndarray         # float64
    close: np.ndarray       # float64 (split- and dividend-adjusted)
    volume: np.ndarray      # float64
    source: str             # "alpaca" or "yfinance"
    start: date             # requested window start (inclusive)
    end: date               # requested window end (exclusive)

    def __post_init__(self) -> None:
        n = len(self.dates)
        for name in OHLCV_FIELDS:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"length mismatch: dates has {n} entries, {name} has {len(arr)}"
                )

    def __len__(self) -> int:
        return len(self.dates)

    def __repr__(self) -> str:
        if len(self) == 0:
            return f"StockReturnTS({self.ticker}, 0 bars)"
        d0 = pd.Timestamp(self.dates[0]).date()
        d1 = pd.Timestamp(self.dates[-1]).date()
        return (
            f"StockReturnTS({self.ticker}, {len(self)} bars, "
            f"{d0} -> {d1}, source={self.source})"
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        ticker: str,
        source: str,
        start: date,
        end: date,
    ) -> "StockReturnTS":
        """Build a StockReturnTS from a DataFrame indexed by date with
        columns open, high, low, close, volume."""
        return cls(
            ticker=ticker,
            dates=df.index.to_numpy(),
            open=df["open"].to_numpy(dtype=float),
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            volume=df["volume"].to_numpy(dtype=float),
            source=source,
            start=start,
            end=end,
        )
