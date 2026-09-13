"""Tests for StockReturnTS container."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.data.stock_return_ts import OHLCV_FIELDS, StockReturnTS


def _make_valid(n: int = 5) -> StockReturnTS:
    return StockReturnTS(
        ticker="TEST",
        dates=pd.date_range("2026-01-05", periods=n, freq="B").to_numpy(),
        open=np.arange(n, dtype=float),
        high=np.arange(n, dtype=float) + 1,
        low=np.arange(n, dtype=float) - 1,
        close=np.arange(n, dtype=float) + 0.5,
        volume=np.full(n, 1_000_000.0),
        source="alpaca",
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
    )


def test_construction_happy_path():
    ts = _make_valid(5)
    assert ts.ticker == "TEST"
    assert len(ts) == 5
    assert ts.source == "alpaca"
    assert ts.close[2] == 2.5


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        StockReturnTS(
            ticker="TEST",
            dates=pd.date_range("2026-01-05", periods=5, freq="B").to_numpy(),
            open=np.zeros(5),
            high=np.zeros(5),
            low=np.zeros(5),
            close=np.zeros(4),  # short by one
            volume=np.zeros(5),
            source="alpaca",
            start=date(2026, 1, 1),
            end=date(2026, 1, 10),
        )


@pytest.mark.parametrize("short_field", OHLCV_FIELDS)
def test_every_ohlcv_field_validated(short_field):
    """Each field in OHLCV_FIELDS must be length-checked."""
    n = 5
    kwargs = dict(
        ticker="TEST",
        dates=pd.date_range("2026-01-05", periods=n, freq="B").to_numpy(),
        open=np.zeros(n),
        high=np.zeros(n),
        low=np.zeros(n),
        close=np.zeros(n),
        volume=np.zeros(n),
        source="alpaca",
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
    )
    kwargs[short_field] = np.zeros(n - 1)
    with pytest.raises(ValueError, match=short_field):
        StockReturnTS(**kwargs)


def test_repr_one_liner():
    ts = _make_valid(5)
    r = repr(ts)
    assert "TEST" in r
    assert "5 bars" in r
    assert "source=alpaca" in r


def test_repr_empty():
    ts = StockReturnTS(
        ticker="EMPTY",
        dates=np.array([], dtype="datetime64[ns]"),
        open=np.array([]),
        high=np.array([]),
        low=np.array([]),
        close=np.array([]),
        volume=np.array([]),
        source="alpaca",
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
    )
    assert "0 bars" in repr(ts)
    assert len(ts) == 0


def test_from_dataframe_round_trip():
    idx = pd.date_range("2026-01-05", periods=3, freq="B", name="date")
    df = pd.DataFrame(
        {
            "open":  [10.0, 11.0, 12.0],
            "high":  [10.5, 11.5, 12.5],
            "low":   [ 9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume":[1e6,  2e6,  3e6],
        },
        index=idx,
    )
    ts = StockReturnTS.from_dataframe(
        df, ticker="META", source="alpaca",
        start=date(2026, 1, 1), end=date(2026, 1, 10),
    )
    assert ts.ticker == "META"
    assert len(ts) == 3
    np.testing.assert_array_equal(ts.close, [10.2, 11.2, 12.2])
    assert ts.close.dtype == np.float64
