"""Tests for the interactive review service. Data fetchers + broker mocked (no network)."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from options_trader import run_daily as rd_mod
from options_trader import manage_positions as mp_mod
from options_trader.data import spot_anchor as anchor_mod
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.execution.broker import AlpacaBroker, ExecutionStatus
from options_trader.forecast.factories import bootstrap_factory
from options_trader.portfolio.constructor import PortfolioCaps
from options_trader.valuation.option_valuer import OptionContract, OptionType
from options_trader.webapp.review_service import build_review_session, execute_item


RUN_DATE = date(2026, 6, 2)
EXPIRY = date(2026, 6, 23)              # ~21 DTE
HELD_SYMBOL = "META260623P00095000"     # OCC: META put, strike 95, expiry 2026-06-23
PERMISSIVE = PortfolioCaps(max_gross_fraction=1.0, hedge_threshold=1.0, max_per_name_fraction=1.0)


def _ts(ticker="META", n=400, start=100.0, drift=-0.0008):
    # gently downward-drifting series → puts look cheap (BUY signals)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(0)
    logret = drift + 0.02 * rng.standard_normal(n)
    close = start * np.exp(np.cumsum(logret))
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": np.full(n, 1e6)},
        index=pd.Index(idx, name="date"),
    )
    return StockReturnTS.from_dataframe(df, ticker=ticker, source="test",
                                        start=date(2024, 1, 2), end=RUN_DATE)


def _chain(spot):
    """Cheap OTM puts → BUY against the down-drift forecast."""
    out = []
    for k in (spot * 0.95, spot * 0.97, spot * 0.99):
        out.append(OptionContract(
            symbol=f"P{k:.0f}", underlying="META", strike=round(k, 0), expiry=EXPIRY,
            option_type=OptionType.PUT, bid=spot * 0.005, ask=spot * 0.006,
        ))
    return out


def _held_quote_chain(ticker, **kw):
    """The live quote for the exact held contract (matched by OCC symbol)."""
    spot = float(_ts(ticker).close[-1])
    return [OptionContract(
        symbol=HELD_SYMBOL, underlying="META", strike=95.0, expiry=EXPIRY,
        option_type=OptionType.PUT, bid=spot * 0.01, ask=spot * 0.012,
    )]


@pytest.fixture
def patched(monkeypatch):
    # entry pass (plan_ticker reaches these via the run_daily module)
    monkeypatch.setattr(rd_mod, "get_history", lambda t: _ts(t))
    monkeypatch.setattr(rd_mod, "_select_expiry", lambda t, s, dte, rd: (EXPIRY, 15))
    monkeypatch.setattr(rd_mod, "get_option_chain",
                        lambda ticker, **kw: _chain(float(_ts(ticker).close[-1])))
    # exit pass (review_position reaches these via the manage_positions module)
    monkeypatch.setattr(mp_mod, "get_history", lambda t: _ts(t))
    monkeypatch.setattr(mp_mod, "get_option_chain", _held_quote_chain)
    # no live price → anchor on last close (no network)
    monkeypatch.setattr(anchor_mod, "get_latest_price", lambda t: None)


def _healthy_account(equity=1_000_000):
    return SimpleNamespace(
        equity=str(equity), cash=str(equity), buying_power=str(equity),
        options_buying_power=str(equity), options_trading_level="3",
        status="ACTIVE", trading_blocked=False,
    )


def _broker(positions=None, orders=None, equity=1_000_000):
    client = MagicMock()
    client.get_account.return_value = _healthy_account(equity)
    client.get_all_positions.return_value = positions or []
    client.get_orders.return_value = orders or []
    return AlpacaBroker(dry_run=True, client=client)


def _option_position(symbol=HELD_SYMBOL, qty=3):
    return SimpleNamespace(
        symbol=symbol, qty=str(qty), side="long", avg_entry_price="1.0",
        market_value="600", unrealized_pl="50", asset_class="us_option",
    )


# ----------------------------- session build -----------------------------

def test_build_session_produces_buys_with_chart_and_budget(patched):
    session = build_review_session(
        _broker(), ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=PERMISSIVE, forecaster_factory=bootstrap_factory,
    )
    p = session.payload
    assert p["buys"], "expected at least one buy candidate"
    b = p["buys"][0]
    assert b["item_id"].startswith("buy:")
    assert b["sizing"]["n_contracts"] >= 1
    assert b["fair_value"] is not None and b["forecast_mean"] is not None
    # faithful chart payload for the decision view
    ch = b["chart"]
    assert ch is not None
    assert len(ch["terminal"]["grid"]) == len(ch["terminal"]["pdf"])
    assert ch["strike"] == b["strike"]
    assert ch["spot"] == b["spot"]
    assert ch["forecast_mean"] == pytest.approx(b["forecast_mean"])
    assert ch["history"]["dates"] and ch["history"]["close"]
    # budget block reflects the 20%-ish gross cap and starts undeployed
    bud = p["budget"]
    assert bud["bankroll"] == 1_000_000
    assert bud["gross_budget"] > 0
    assert bud["deployed_so_far"] == 0.0
    assert bud["remaining"] == pytest.approx(bud["gross_budget"] - bud["current_premium"])


def test_build_session_reviews_held_positions_as_sells(patched):
    broker = _broker(positions=[_option_position()])
    session = build_review_session(
        broker, ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=PERMISSIVE, forecaster_factory=bootstrap_factory,
    )
    p = session.payload
    assert len(p["sells"]) == 1
    s = p["sells"][0]
    assert s["item_id"] == f"sell:{HELD_SYMBOL}"
    assert s["underlying"] == "META" and s["option_type"] == "put"
    assert s["qty"] == 3
    assert s["bid"] is not None and s["fair_value"] is not None
    assert s["action"] in ("SELL", "HOLD")
    assert s["chart"] is not None  # decision view available


def test_held_put_excluded_from_buys(patched):
    # holding a META put → the down-drift chain's only BUYs (puts) are suppressed
    broker = _broker(positions=[_option_position()])
    session = build_review_session(
        broker, ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=PERMISSIVE, forecaster_factory=bootstrap_factory,
    )
    assert session.payload["buys"] == []


# ----------------------------- execution -----------------------------

def test_execute_buy_dispatches_and_updates_budget(patched):
    broker = _broker()
    session = build_review_session(
        broker, ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=PERMISSIVE, forecaster_factory=bootstrap_factory,
    )
    session._broker = broker
    item = session.payload["buys"][0]
    cost = item["sizing"]["cost"]
    res = execute_item(session, item["item_id"])
    assert res["order"]["status"] == ExecutionStatus.DRY_RUN.value
    assert res["order"]["side"] == "buy"
    assert res["budget"]["deployed_so_far"] == pytest.approx(cost)
    assert session.deployed_so_far == pytest.approx(cost)


def test_execute_sell_calls_submit_sell(patched):
    broker = _broker(positions=[_option_position()])
    session = build_review_session(
        broker, ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=PERMISSIVE, forecaster_factory=bootstrap_factory,
    )
    session._broker = broker
    spy = MagicMock(wraps=broker.submit_sell)
    broker.submit_sell = spy
    item = session.payload["sells"][0]
    res = execute_item(session, item["item_id"])
    spy.assert_called_once()
    assert spy.call_args.args[0] == HELD_SYMBOL          # symbol
    assert res["order"]["side"] == "sell"
    assert res["order"]["status"] == ExecutionStatus.DRY_RUN.value


def test_buy_over_budget_blocks_then_overrides(patched):
    # tiny gross budget → the candidate exceeds it: blocked unless overridden
    tight = PortfolioCaps(max_gross_fraction=0.0001, hedge_threshold=1.0, max_per_name_fraction=1.0)
    broker = _broker()
    session = build_review_session(
        broker, ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=tight, forecaster_factory=bootstrap_factory,
    )
    session._broker = broker
    item = session.payload["buys"][0]
    blocked = execute_item(session, item["item_id"])
    assert blocked.get("blocked") == "over_budget"
    assert session.deployed_so_far == 0.0          # nothing deployed
    forced = execute_item(session, item["item_id"], override=True)
    assert forced["order"]["status"] == ExecutionStatus.DRY_RUN.value
    assert session.deployed_so_far > 0.0


def test_execute_unknown_item_id(patched):
    session = build_review_session(
        _broker(), ["META"], run_date=RUN_DATE, bankroll=1_000_000,
        caps=PERMISSIVE, forecaster_factory=bootstrap_factory,
    )
    session._broker = session.__dict__.get("_broker")  # none attached
    res = execute_item(session, "buy:NOPE")
    assert "error" in res
