"""Tests for the options-chain layer. All network calls are mocked."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import ContractType

from options_trader.data import options_chain
from options_trader.data.history import CredentialsError
from options_trader.data.options_chain import (
    OptionsChainError,
    OptionsDataUnavailableError,
    _build_contracts,
    _to_option_type,
    get_option_chain,
)
from options_trader.valuation.option_valuer import OptionContract, OptionType


# ---------- synthetic Alpaca-shaped objects ----------

def _raw_contract(symbol, strike, expiry, ctype=ContractType.CALL, tradable=True, oi=123):
    return SimpleNamespace(
        symbol=symbol,
        underlying_symbol="META",
        strike_price=str(strike),       # Alpaca returns strike as a string
        expiration_date=expiry,
        type=ctype,
        tradable=tradable,
        open_interest=str(oi) if oi is not None else None,
    )


def _snapshot(symbol, bid, ask, iv=0.45):
    quote = SimpleNamespace(bid_price=bid, ask_price=ask)
    return SimpleNamespace(symbol=symbol, latest_quote=quote, implied_volatility=iv)


# ---------- _to_option_type ----------

def test_to_option_type_from_enum():
    assert _to_option_type(ContractType.CALL) == OptionType.CALL
    assert _to_option_type(ContractType.PUT) == OptionType.PUT


def test_to_option_type_from_string():
    assert _to_option_type("call") == OptionType.CALL
    assert _to_option_type("PUT") == OptionType.PUT


# ---------- _build_contracts (pure merge) ----------

def test_build_merges_quote_and_maps_fields():
    raw = [_raw_contract("META260116C00500000", 500.0, date(2026, 1, 16), oi=42)]
    snaps = {"META260116C00500000": _snapshot("META260116C00500000", bid=12.0, ask=12.5, iv=0.4)}
    out = _build_contracts(raw, snaps, include_untradable=False, underlying="META")
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, OptionContract)
    assert c.symbol == "META260116C00500000"
    assert c.underlying == "META"
    assert c.strike == 500.0 and isinstance(c.strike, float)
    assert c.expiry == date(2026, 1, 16)
    assert c.option_type == OptionType.CALL
    assert c.bid == 12.0 and c.ask == 12.5
    assert c.implied_vol == 0.4
    assert c.open_interest == 42 and isinstance(c.open_interest, int)


def test_build_contract_without_snapshot_has_no_quote():
    raw = [_raw_contract("X", 500.0, date(2026, 1, 16))]
    out = _build_contracts(raw, {}, include_untradable=False, underlying="META")
    assert out[0].bid is None and out[0].ask is None and out[0].implied_vol is None


def test_build_handles_snapshot_missing_latest_quote():
    raw = [_raw_contract("X", 500.0, date(2026, 1, 16))]
    snap = SimpleNamespace(symbol="X", latest_quote=None, implied_volatility=0.5)
    out = _build_contracts(raw, {"X": snap}, include_untradable=False, underlying="META")
    assert out[0].bid is None and out[0].ask is None
    assert out[0].implied_vol == 0.5  # IV still captured


def test_build_filters_untradable_by_default():
    raw = [
        _raw_contract("OK", 500.0, date(2026, 1, 16), tradable=True),
        _raw_contract("NO", 510.0, date(2026, 1, 16), tradable=False),
    ]
    out = _build_contracts(raw, {}, include_untradable=False, underlying="META")
    assert [c.symbol for c in out] == ["OK"]


def test_build_include_untradable_keeps_all():
    raw = [
        _raw_contract("OK", 500.0, date(2026, 1, 16), tradable=True),
        _raw_contract("NO", 510.0, date(2026, 1, 16), tradable=False),
    ]
    out = _build_contracts(raw, {}, include_untradable=True, underlying="META")
    assert len(out) == 2


def test_build_sorts_by_expiry_then_strike_then_type():
    raw = [
        _raw_contract("c", 500.0, date(2026, 2, 20), ContractType.CALL),
        _raw_contract("a", 500.0, date(2026, 1, 16), ContractType.CALL),
        _raw_contract("b", 510.0, date(2026, 1, 16), ContractType.CALL),
    ]
    out = _build_contracts(raw, {}, include_untradable=False, underlying="META")
    assert [c.symbol for c in out] == ["a", "b", "c"]


def test_build_none_open_interest():
    raw = [_raw_contract("X", 500.0, date(2026, 1, 16), oi=None)]
    out = _build_contracts(raw, {}, include_untradable=False, underlying="META")
    assert out[0].open_interest is None


# ---------- get_option_chain orchestration ----------

def test_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(CredentialsError):
        get_option_chain("META", expiration_lte=date(2026, 1, 16))


def test_happy_path_merges(monkeypatch):
    raw = [_raw_contract("META_C", 500.0, date(2026, 1, 16))]
    snaps = {"META_C": _snapshot("META_C", bid=10.0, ask=10.4)}
    with patch.object(options_chain, "_fetch_contracts", return_value=raw), \
         patch.object(options_chain, "_fetch_snapshots", return_value=snaps):
        out = get_option_chain("META")
    assert len(out) == 1
    assert out[0].ask == 10.4


def test_empty_contracts_raises(monkeypatch):
    with patch.object(options_chain, "_fetch_contracts", return_value=[]):
        with pytest.raises(OptionsDataUnavailableError, match="No tradable"):
            get_option_chain("META")


def test_contracts_apierror_raises_data_unavailable():
    with patch.object(options_chain, "_fetch_contracts", side_effect=APIError("boom")):
        with pytest.raises(OptionsDataUnavailableError, match="Failed to fetch"):
            get_option_chain("META")


def test_snapshot_failure_returns_quoteless_contracts():
    """Quotes are best-effort: a snapshot failure must not lose the contracts."""
    raw = [_raw_contract("META_C", 500.0, date(2026, 1, 16))]
    with patch.object(options_chain, "_fetch_contracts", return_value=raw), \
         patch.object(options_chain, "_fetch_snapshots", side_effect=APIError("quotes down")):
        out = get_option_chain("META")
    assert len(out) == 1
    assert out[0].bid is None and out[0].ask is None


def test_credentials_error_in_contracts_propagates():
    with patch.object(options_chain, "_fetch_contracts", side_effect=CredentialsError("nope")):
        with pytest.raises(CredentialsError):
            get_option_chain("META")


def test_invalid_feed_raises():
    with pytest.raises(OptionsChainError, match="feed must be"):
        get_option_chain("META", feed="bogus")


# ---------- Alpaca symbol-format translation (dash-class tickers, e.g. BRK-B) ----------

def test_build_contracts_stamps_canonical_underlying_not_alpacas():
    """Alpaca reports its OWN dot-format underlying_symbol on each contract; the
    caller-supplied canonical (dash-format) ticker must win, so downstream code
    (portfolio grouping, held-type guards) stays keyed consistently."""
    raw = [SimpleNamespace(
        symbol="BRKB260116C00500000", underlying_symbol="BRK.B",
        strike_price="500.0", expiration_date=date(2026, 1, 16),
        type=ContractType.CALL, tradable=True, open_interest="10",
    )]
    out = _build_contracts(raw, {}, include_untradable=False, underlying="BRK-B")
    assert out[0].underlying == "BRK-B"


def test_fetch_contracts_translates_dash_class_ticker(monkeypatch):
    from options_trader.data import options_chain as oc_mod

    captured = {}

    class _FakeTradingClient:
        def __init__(self, *a, **kw):
            pass

        def get_option_contracts(self, req):
            captured["underlying_symbols"] = req.underlying_symbols
            return SimpleNamespace(option_contracts=[], next_page_token=None)

    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")
    monkeypatch.setattr(oc_mod, "TradingClient", _FakeTradingClient)
    oc_mod._fetch_contracts("BRK-B", None, None, None, None, None)
    assert captured["underlying_symbols"] == ["BRK.B"]


def test_fetch_snapshots_translates_dash_class_ticker(monkeypatch):
    from alpaca.data.enums import OptionsFeed

    from options_trader.data import options_chain as oc_mod

    captured = {}

    class _FakeOptionClient:
        def __init__(self, *a, **kw):
            pass

        def get_option_chain(self, req):
            captured["underlying_symbol"] = req.underlying_symbol
            return {}

    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")
    monkeypatch.setattr(oc_mod, "OptionHistoricalDataClient", _FakeOptionClient)
    oc_mod._fetch_snapshots("BRK-B", None, None, None, None, None, OptionsFeed.INDICATIVE)
    assert captured["underlying_symbol"] == "BRK.B"


def test_get_option_chain_end_to_end_stamps_canonical_underlying():
    """Full orchestration: even for a dash-class ticker, the returned contracts
    carry the canonical ticker the caller asked with."""
    raw = [_raw_contract("BRKB260116C00500000", 500.0, date(2026, 1, 16))]
    with patch.object(options_chain, "_fetch_contracts", return_value=raw), \
         patch.object(options_chain, "_fetch_snapshots", return_value={}):
        out = get_option_chain("BRK-B")
    assert out[0].underlying == "BRK-B"
