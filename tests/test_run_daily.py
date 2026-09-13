"""Orchestration tests for run_daily. Data fetchers + broker are mocked (no network)."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from options_trader import run_daily as rd_mod
from options_trader.data import spot_anchor as anchor_mod
from options_trader.forecast.factories import bootstrap_factory
from options_trader.run_daily import run_daily, plan_ticker
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.execution.broker import AccountSnapshot, AlpacaBroker, ExecutionStatus
from options_trader.valuation.option_valuer import OptionContract, OptionType, OptionValuer
from options_trader.sizing.kelly import KellySizer
from options_trader.portfolio.constructor import PortfolioCaps

PERMISSIVE = PortfolioCaps(max_gross_fraction=1.0, hedge_threshold=1.0, max_per_name_fraction=1.0)


RUN_DATE = date(2026, 6, 2)
EXPIRY = date(2026, 6, 23)  # ~21 DTE


def _ts(ticker="META", n=400, start=100.0, drift=-0.0008):
    # gently downward-drifting series → puts will look cheap (BUY signals)
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
    """A small chain: OTM puts (cheap → BUY against a down-drift forecast)."""
    out = []
    for k in (spot * 0.95, spot * 0.97, spot * 0.99):
        out.append(OptionContract(
            symbol=f"P{k:.0f}", underlying="X", strike=round(k, 0), expiry=EXPIRY,
            option_type=OptionType.PUT, bid=spot * 0.005, ask=spot * 0.006,
        ))
    return out


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(rd_mod, "get_history", lambda t: _ts(t))
    # default: no live quote → forecast anchors on the last close (no network)
    monkeypatch.setattr(anchor_mod, "get_latest_price", lambda t: None)
    monkeypatch.setattr(rd_mod, "_select_expiry", lambda t, s, dte, rd: (EXPIRY, 15))
    monkeypatch.setattr(rd_mod, "get_option_chain",
                        lambda ticker, **kw: _chain(float(_ts(ticker).close[-1])))


def _healthy_account(equity=1_000_000):
    return SimpleNamespace(
        equity=str(equity), cash=str(equity), buying_power=str(equity),
        options_buying_power=str(equity), options_trading_level="3",
        status="ACTIVE", trading_blocked=False,
    )


def _dry_broker(equity=1_000_000):
    # dry-run still reads the account for guards, so give it a healthy mock
    client = MagicMock()
    client.get_account.return_value = _healthy_account(equity)
    return AlpacaBroker(dry_run=True, client=client)


# ---------- plan_ticker ----------

def test_plan_produces_sized_candidates(patched):
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
    )
    assert plan.error is None
    assert plan.spot is not None and plan.forecast_mean is not None
    assert len(plan.candidates) >= 1
    # every candidate carries valuation, sizing, and a drift/shape decomposition
    c = plan.candidates[0]
    assert c.sizing.n_contracts >= 1
    assert c.decomposition.total_edge is not None


def test_spot_anchored_on_live_price(patched, monkeypatch):
    # live price overrides the last daily close as the forecast anchor
    last_close = float(_ts("META").close[-1])
    live = last_close * 1.05
    monkeypatch.setattr(anchor_mod, "get_latest_price", lambda t: live)
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
    )
    assert plan.spot == pytest.approx(live)
    assert plan.spot != pytest.approx(last_close)
    assert plan.spot_source == "live"


def test_spot_falls_back_to_close_when_no_live_price(patched):
    # patched fixture returns None from get_latest_price → fall back to last close
    last_close = float(_ts("META").close[-1])
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
    )
    assert plan.spot == pytest.approx(last_close)
    assert plan.spot_source == "last_close"


def test_custom_spot_anchor_is_injectable(patched):
    # plan_ticker accepts an injected anchor source (parity with backtest/anchor.py)
    from options_trader.data.spot_anchor import AnchoredSpot
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
        spot_anchor=lambda t, ts: AnchoredSpot(spot=123.45, source="custom"),
    )
    assert plan.spot == pytest.approx(123.45)
    assert plan.spot_source == "custom"


def _mixed_chain(spot):
    """Cheap calls AND cheap puts (ask ~0) so every contract is BUY-rated."""
    out = []
    for k in (spot * 0.97, spot * 0.99):
        out.append(OptionContract(
            symbol=f"P{k:.0f}", underlying="X", strike=round(k, 0), expiry=EXPIRY,
            option_type=OptionType.PUT, bid=0.01, ask=0.02))
    for k in (spot * 1.01, spot * 1.03):
        out.append(OptionContract(
            symbol=f"C{k:.0f}", underlying="X", strike=round(k, 0), expiry=EXPIRY,
            option_type=OptionType.CALL, bid=0.01, ask=0.02))
    return out


def test_one_call_and_one_put_max_per_ticker(patched, monkeypatch):
    # mixed chain with 2 calls + 2 puts → keep exactly one of each type
    monkeypatch.setattr(rd_mod, "get_option_chain",
                        lambda ticker, **kw: _mixed_chain(float(_ts(ticker).close[-1])))
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
    )
    types = [c.valuation.contract.option_type for c in plan.candidates]
    assert len(plan.candidates) == 2
    assert set(types) == {OptionType.CALL, OptionType.PUT}


def test_best_of_each_type_kept(patched):
    # puts-only chain (3 strikes) collapses to the single best put by log growth
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
    )
    assert len(plan.candidates) == 1
    assert plan.candidates[0].valuation.contract.option_type == OptionType.PUT


def test_held_type_excluded_from_plan(patched):
    # already holding a put on this name → no new put candidate is planned
    plan = plan_ticker(
        "META", run_date=RUN_DATE, bankroll=1_000_000,
        valuer=OptionValuer(0.04), sizer=KellySizer(0.25, 0.10),
        held_types=frozenset({OptionType.PUT}),
    )
    assert plan.candidates == []


def test_held_positions_block_same_type_in_run_daily(patched):
    # broker reports an open META put → run_daily must not plan another META put
    client = MagicMock()
    client.get_account.return_value = _healthy_account()
    client.get_all_positions.return_value = [
        SimpleNamespace(symbol="META260623P00095000", qty="2", side="long",
                        avg_entry_price="1.0", market_value="200", unrealized_pl="0",
                        asset_class="us_option"),
    ]
    broker = AlpacaBroker(dry_run=True, client=client)
    result = run_daily(["META"], broker, bankroll=1_000_000, caps=PERMISSIVE,
                       forecaster_factory=bootstrap_factory, skip_manage=True)
    # the only BUY signals on the down-drift chain are puts, and puts are held → none
    assert result.all_candidates == []


def test_resting_buy_order_blocks_same_type_in_run_daily(patched):
    # a resting (unfilled) BUY put on META blocks planning another META put
    client = MagicMock()
    client.get_account.return_value = _healthy_account()
    client.get_all_positions.return_value = []
    client.get_orders.return_value = [
        SimpleNamespace(symbol="META260623P00095000", side="buy", qty="2",
                        limit_price="1.0", status="new", id="o1"),
    ]
    broker = AlpacaBroker(dry_run=True, client=client)
    result = run_daily(["META"], broker, bankroll=1_000_000, caps=PERMISSIVE,
                       forecaster_factory=bootstrap_factory, skip_manage=True)
    assert result.all_candidates == []


def test_committed_types_counts_positions_and_buys_not_sells():
    from options_trader.run_daily import _committed_types_by_underlying
    from options_trader.execution.broker import OrderSnapshot, PositionSnapshot
    positions = [PositionSnapshot(symbol="AVGO260702C00405000", qty=3, side="long",
                                  avg_entry_price=14, market_value=4200, unrealized_pl=0,
                                  asset_class="us_option")]
    orders = [
        OrderSnapshot(symbol="GOOGL260626C00380000", side="buy", qty=6, limit_price=5.0,
                      status="new", order_id="o1"),
        OrderSnapshot(symbol="NEM260626P00100000", side="sell", qty=4, limit_price=0.2,
                      status="new", order_id="o2"),
    ]
    committed = _committed_types_by_underlying(positions, orders)
    assert committed["AVGO"] == frozenset({OptionType.CALL})   # position
    assert committed["GOOGL"] == frozenset({OptionType.CALL})  # resting BUY
    assert "NEM" not in committed   # resting SELL is a pending close, not a new long


# ---------- expiry selection ----------

def test_expiry_window_spans_a_monthly_cycle():
    # The probe window must reach target+35 days so a full monthly cycle is
    # always covered — otherwise a monthly-only name (no weeklies) whose sole
    # expiry lands between two monthlies would find nothing.
    from options_trader.run_daily import _expiry_window
    run_date = date(2026, 6, 9)
    lo, hi = _expiry_window(target_dte=21, run_date=run_date)
    # lower bound conservative (target-10), upper bound spans a monthly cycle (target+35)
    assert lo == date.fromordinal(run_date.toordinal() + 11)
    assert hi == date.fromordinal(run_date.toordinal() + 21 + 35)


def test_select_expiry_picks_closest_to_target():
    # given several expiries, the one closest to target_dte wins (window width is moot)
    from options_trader.run_daily import _select_expiry
    run_date = date(2026, 6, 9)

    chain = [
        OptionContract(symbol="A", underlying="X", strike=100.0,
                       expiry=date(2026, 6, 26), option_type=OptionType.CALL, bid=1, ask=1.1),
        OptionContract(symbol="B", underlying="X", strike=100.0,
                       expiry=date(2026, 7, 2), option_type=OptionType.CALL, bid=1, ask=1.1),
        OptionContract(symbol="C", underlying="X", strike=100.0,
                       expiry=date(2026, 7, 17), option_type=OptionType.CALL, bid=1, ask=1.1),
    ]
    # target day = June 30; July 2 (DTE 23) is closest
    expiry, _ = _select_expiry("X", chain, target_dte=21, run_date=run_date)
    assert expiry == date(2026, 7, 2)


# ---------- error isolation ----------

def test_history_error_isolated(monkeypatch):
    def fake_history(t):
        if t == "BAD":
            from options_trader.data.history import DataUnavailableError
            raise DataUnavailableError("no data")
        return _ts(t)
    monkeypatch.setattr(rd_mod, "get_history", fake_history)
    monkeypatch.setattr(anchor_mod, "get_latest_price", lambda t: None)
    monkeypatch.setattr(rd_mod, "_select_expiry", lambda t, s, dte, rd: (EXPIRY, 15))
    monkeypatch.setattr(rd_mod, "get_option_chain",
                        lambda ticker, **kw: _chain(float(_ts(ticker).close[-1])))

    result = run_daily(["META", "BAD", "AAPL"], _dry_broker(), run_date=RUN_DATE,
                       bankroll=1_000_000, forecaster_factory=bootstrap_factory, skip_manage=True)
    by_ticker = {p.ticker: p for p in result.plans}
    assert by_ticker["BAD"].error is not None
    assert by_ticker["META"].error is None and by_ticker["AAPL"].error is None
    assert len(result.plans) == 3


# ---------- run_daily end-to-end (dry-run) ----------

def test_run_daily_dry_run_executes_dry(patched):
    # permissive caps so both candidates fund (isolates orchestration from the caps)
    result = run_daily(["META", "AAPL"], _dry_broker(), run_date=RUN_DATE, bankroll=1_000_000,
                       caps=PERMISSIVE, forecaster_factory=bootstrap_factory, skip_manage=True)
    assert result.dry_run is True
    cands = result.all_candidates
    assert len(cands) == 2  # puts-only chain → best put per ticker = 1 each
    assert all(c.allocated for c in cands)
    assert all(c.order.status == ExecutionStatus.DRY_RUN for c in cands)
    assert result.total_cost > 0
    assert result.aggregate_deployed_fraction > 0


def test_run_daily_rejected_candidates_not_executed(patched):
    # tiny gross budget → at most one small position funds; the rest are rejected
    # and must NOT be executed (order stays None).
    tight = PortfolioCaps(max_gross_fraction=0.001, hedge_threshold=1.0, max_per_name_fraction=1.0)
    result = run_daily(["META", "AAPL", "NVDA"], _dry_broker(), bankroll=1_000_000, caps=tight,
                       forecaster_factory=bootstrap_factory, skip_manage=True)
    for c in result.all_candidates:
        if c.allocated:
            assert c.order is not None and c.order.status == ExecutionStatus.DRY_RUN
        else:
            assert c.order is None and c.alloc_reason is not None
    # nothing deployed beyond the gross budget
    assert result.total_cost <= result.allocation.gross_budget + 1e-6


def test_run_daily_bankroll_from_account(monkeypatch, patched):
    # bankroll=None → pulled from the broker account snapshot
    broker = _dry_broker()
    snap = AccountSnapshot(equity=250_000, cash=250_000, buying_power=250_000,
                           options_buying_power=250_000, options_trading_level=3,
                           status="ACTIVE", trading_blocked=False)
    monkeypatch.setattr(broker, "get_account_snapshot", lambda: snap)
    result = run_daily(["META"], broker, bankroll=None, forecaster_factory=bootstrap_factory,
                       skip_manage=True)
    assert result.bankroll == 250_000


def test_default_forecaster_factory_is_blend():
    """Regression (S18): the production default is the 50/50 BlendFactory of
    event-bootstrap ⊕ event-conditioned OIB (switched from EventBootstrapFactory).

    The blend beat event-bootstrap on the S18 1yr/40-name backtest (+0.0107, p≈6e-14);
    switched per user direction. The two sub-factories must be EventBootstrapFactory and
    an event-conditioned OptionImpliedBetaFactory at equal weight; must NOT silently
    revert. See docs/results.md S18 + docs/decisions.md.
    """
    from options_trader.forecast.factories import (
        BlendFactory, EventBootstrapFactory, OptionImpliedBetaFactory,
    )
    factory = rd_mod._default_forecaster_factory(event_window=0)
    assert isinstance(factory, BlendFactory)
    assert len(factory.factories) == 2
    assert isinstance(factory.factories[0], EventBootstrapFactory)
    assert isinstance(factory.factories[1], OptionImpliedBetaFactory)
    # OIB arm must be event-conditioned (matches the backtested arm).
    assert factory.factories[1].event_source is not None
    assert tuple(factory.weights) == (0.5, 0.5)
