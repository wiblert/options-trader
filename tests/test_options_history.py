"""Tests for data/options_history.py. Network is mocked / stubbed."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from options_trader.data.history import CredentialsError
from options_trader.data.options_history import (
    AlpacaOptionBarsProvider,
    OptionBar,
    OptionBarsProvider,
    OptionsHistoryError,
    _bars_from_alpaca_df,
    get_option_bars_eod,
)


# ---------- pure parse core ----------

def _alpaca_df(rows):
    """rows: list of (symbol, timestamp, o,h,l,c,v) -> MultiIndex frame like alpaca .df"""
    idx = pd.MultiIndex.from_tuples(
        [(s, pd.Timestamp(t, tz="UTC")) for s, t, *_ in rows],
        names=["symbol", "timestamp"],
    )
    data = {
        "open": [r[2] for r in rows], "high": [r[3] for r in rows],
        "low": [r[4] for r in rows], "close": [r[5] for r in rows],
        "volume": [r[6] for r in rows],
    }
    return pd.DataFrame(data, index=idx)


def test_parse_groups_by_symbol_and_sorts():
    df = _alpaca_df([
        ("SPY260116C00500000", "2026-01-05 21:00", 10, 11, 9, 10.5, 100),
        ("SPY260116C00500000", "2026-01-02 21:00", 9, 10, 8, 9.5, 80),
        ("IWM260116C00200000", "2026-01-05 21:00", 5, 6, 4, 5.5, 50),
    ])
    out = _bars_from_alpaca_df(df)
    assert set(out) == {"SPY260116C00500000", "IWM260116C00200000"}
    spy = out["SPY260116C00500000"]
    assert len(spy) == 2
    assert [b.bar_date for b in spy] == sorted(b.bar_date for b in spy)  # oldest first
    assert isinstance(spy[0], OptionBar)
    assert spy[0].close == 9.5


def test_parse_empty_frame_returns_empty_dict():
    assert _bars_from_alpaca_df(None) == {}
    assert _bars_from_alpaca_df(pd.DataFrame()) == {}


def test_parse_rejects_unexpected_columns():
    bad = pd.DataFrame({"foo": [1]})
    with pytest.raises(OptionsHistoryError):
        _bars_from_alpaca_df(bad)


# ---------- AlpacaOptionBarsProvider with a stub client ----------

class _StubClient:
    def __init__(self, df):
        self._df = df
        self.calls = []

    def get_option_bars(self, req):
        self.calls.append(req)
        return SimpleNamespace(df=self._df)


def test_provider_returns_bars_via_injected_client():
    df = _alpaca_df([("SPY260116C00500000", "2026-01-05 21:00", 10, 11, 9, 10.5, 100)])
    provider = AlpacaOptionBarsProvider(client=_StubClient(df))
    out = provider.get_eod_bars(["SPY260116C00500000"], date(2026, 1, 1), date(2026, 1, 6))
    assert "SPY260116C00500000" in out and len(out["SPY260116C00500000"]) == 1


def test_provider_satisfies_protocol():
    assert isinstance(AlpacaOptionBarsProvider(client=_StubClient(pd.DataFrame())),
                      OptionBarsProvider)


def test_provider_empty_symbols_short_circuits():
    provider = AlpacaOptionBarsProvider(client=_StubClient(pd.DataFrame()))
    assert provider.get_eod_bars([], date(2026, 1, 1), date(2026, 1, 6)) == {}


def test_provider_rejects_reversed_window():
    provider = AlpacaOptionBarsProvider(client=_StubClient(pd.DataFrame()))
    with pytest.raises(OptionsHistoryError):
        provider.get_eod_bars(["X"], date(2026, 1, 6), date(2026, 1, 1))


def test_get_option_bars_eod_uses_injected_provider():
    class _P:
        def get_eod_bars(self, symbols, start, end):
            return {"X": [OptionBar("X", date(2026, 1, 2), 1, 1, 1, 1, 1)]}
    out = get_option_bars_eod(["X"], date(2026, 1, 1), date(2026, 1, 3), provider=_P())
    assert out["X"][0].symbol == "X"


# ---------- list_contracts (expired-contract discovery) ----------

def test_list_contracts_parses_and_filters(monkeypatch):
    from options_trader.data.options_history import ContractRef, list_contracts

    class _TC:
        def get_option_contracts(self, req):
            # echo the requested type; two pages to exercise pagination
            ctype = req.type.value if req.type is not None else "call"
            if req.page_token is None:
                cs = [SimpleNamespace(symbol="SPY250417C00500000", strike_price="500",
                                      expiration_date=date(2025, 4, 17), type=ctype)]
                return SimpleNamespace(option_contracts=cs, next_page_token="p2")
            cs = [SimpleNamespace(symbol="SPY250417C00510000", strike_price="510",
                                  expiration_date=date(2025, 4, 17), type=ctype)]
            return SimpleNamespace(option_contracts=cs, next_page_token=None)

    out = list_contracts("SPY", expiration_gte=date(2025, 4, 1),
                         expiration_lte=date(2025, 4, 30), option_type="call",
                         status="inactive", trading_client=_TC())
    assert len(out) == 2 and all(isinstance(c, ContractRef) for c in out)
    assert out[0].strike == 500.0 and out[0].option_type == "call"


def test_list_contracts_rejects_bad_status():
    from options_trader.data.options_history import list_contracts
    with pytest.raises(ValueError):
        list_contracts("SPY", expiration_gte=date(2025, 4, 1),
                       expiration_lte=date(2025, 4, 30), status="bogus",
                       trading_client=object())


def test_default_provider_without_creds_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(CredentialsError):
        AlpacaOptionBarsProvider().get_eod_bars(["X"], date(2026, 1, 1), date(2026, 1, 3))
