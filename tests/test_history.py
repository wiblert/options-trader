"""Tests for get_history orchestration. Network calls are mocked."""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from alpaca.common.exceptions import APIError

from options_trader.data import history
from options_trader.data.history import (
    CredentialsError,
    DataUnavailableError,
    HistoryError,
    get_history,
)
from options_trader.data.stock_return_ts import StockReturnTS


# ---------- helpers ----------

def _fake_bars(n_days: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=n_days, freq="B", name="date")
    return pd.DataFrame(
        {
            "open":   np.linspace(100, 200, n_days),
            "high":   np.linspace(101, 201, n_days),
            "low":    np.linspace( 99, 199, n_days),
            "close":  np.linspace(100, 200, n_days),
            "volume": np.full(n_days, 1e6),
        },
        index=idx,
    )


# ---------- input validation ----------

def test_start_after_end_raises():
    with pytest.raises(HistoryError, match="must be before"):
        get_history("META", start=date(2026, 6, 1), end=date(2026, 5, 1))


def test_missing_credentials_raises_and_does_not_fall_back(monkeypatch):
    """Critical invariant: CredentialsError must propagate, NOT silently demote to yfinance."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with patch.object(history, "_from_yfinance") as yf_mock:
        with pytest.raises(CredentialsError):
            get_history("META", start=date(2025, 1, 1), end=date(2025, 6, 1))
    yf_mock.assert_not_called()


# ---------- happy path: Alpaca succeeds ----------

def test_alpaca_success_returns_stockreturnts(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    with patch.object(history, "_from_alpaca", return_value=_fake_bars(200)):
        ts = get_history("META", start=date(2025, 1, 1), end=date(2025, 12, 31))

    assert isinstance(ts, StockReturnTS)
    assert ts.source == "alpaca"
    assert len(ts) == 200


# ---------- fallback on APIError ----------

def test_alpaca_apierror_falls_back_to_yfinance(monkeypatch, caplog):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    with caplog.at_level("WARNING", logger="options_trader.data.history"):
        with patch.object(history, "_from_alpaca", side_effect=APIError("rate limited")):
            with patch.object(history, "_from_yfinance", return_value=_fake_bars(200)):
                ts = get_history("META", start=date(2025, 1, 1), end=date(2025, 12, 31))

    assert ts.source == "yfinance"
    assert any("Alpaca APIError" in r.message for r in caplog.records)


def test_alpaca_timeout_falls_back_to_yfinance(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    with patch.object(history, "_from_alpaca", side_effect=TimeoutError("hung")):
        with patch.object(history, "_from_yfinance", return_value=_fake_bars(200)):
            ts = get_history("META", start=date(2025, 1, 1), end=date(2025, 12, 31))

    assert ts.source == "yfinance"


def test_short_alpaca_response_falls_back(monkeypatch):
    """If Alpaca returns < MIN_BARS_THRESHOLD bars, we should try yfinance."""
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    with patch.object(history, "_from_alpaca", return_value=_fake_bars(20)):
        with patch.object(history, "_from_yfinance", return_value=_fake_bars(200)):
            ts = get_history("META", start=date(2025, 1, 1), end=date(2025, 12, 31))

    assert ts.source == "yfinance"


# ---------- both sources fail ----------

def test_both_sources_empty_raises_data_unavailable(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    with patch.object(history, "_from_alpaca", return_value=pd.DataFrame()):
        with patch.object(history, "_from_yfinance", return_value=pd.DataFrame()):
            with pytest.raises(DataUnavailableError):
                get_history("META", start=date(2025, 1, 1), end=date(2025, 12, 31))


def test_yfinance_also_fails_raises_data_unavailable(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    with patch.object(history, "_from_alpaca", side_effect=APIError("down")):
        with patch.object(history, "_from_yfinance", side_effect=TimeoutError("also down")):
            with pytest.raises(DataUnavailableError, match="Both sources failed"):
                get_history("META", start=date(2025, 1, 1), end=date(2025, 12, 31))
