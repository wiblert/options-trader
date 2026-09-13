"""Tests for the exit-pass decision rule + orchestration (no network)."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from options_trader.data.history import HistoryError
from options_trader.execution.broker import AlpacaBroker, ExecutionStatus
from options_trader import manage_positions as mp
from options_trader.manage_positions import PositionReview, decide_action, manage_positions


# ---------- the pure hold/sell rule ----------

def test_sell_when_fair_far_below_bid():
    # bid 10, fair 8 -> 20% below -> sell at 15% threshold
    action, gap, _ = decide_action(fair_value=8.0, bid=10.0, sell_threshold=0.15)
    assert action == "SELL"
    assert round(gap, 4) == 0.2


def test_hold_when_gap_under_threshold():
    action, gap, _ = decide_action(fair_value=9.5, bid=10.0, sell_threshold=0.15)
    assert action == "HOLD"
    assert round(gap, 4) == 0.05


def test_hold_when_fair_above_bid():
    # our value exceeds the bid -> market undervalues it -> hold (negative gap)
    action, gap, _ = decide_action(fair_value=12.0, bid=10.0, sell_threshold=0.15)
    assert action == "HOLD"
    assert gap < 0


def test_hold_when_no_bid():
    action, gap, reason = decide_action(fair_value=8.0, bid=None, sell_threshold=0.15)
    assert action == "HOLD"
    assert gap is None
    assert "no live bid" in reason


def test_threshold_boundary_is_inclusive():
    action, _, _ = decide_action(fair_value=8.5, bid=10.0, sell_threshold=0.15)
    assert action == "SELL"  # exactly 15% below


# ---------- orchestration: only option positions, dry-run safety ----------

def _mock_client_with_positions(positions):
    c = MagicMock()
    c.get_all_positions.return_value = positions
    return c


def test_manage_only_reviews_option_positions():
    positions = [
        SimpleNamespace(symbol="AAPL", qty="100", side="long", avg_entry_price="150",
                        market_value="15000", unrealized_pl="0", asset_class="us_equity"),
    ]
    broker = AlpacaBroker(client=_mock_client_with_positions(positions), dry_run=True)
    # forecaster_factory is irrelevant: the only position is equity, so it's filtered out.
    result = manage_positions(broker, run_date=date(2026, 6, 9),
                              forecaster_factory=lambda ctx: None)
    assert result.reviews == []
    assert result.sells == []


def test_manage_dry_run_isolates_errors_and_never_submits():
    # An option position that would be reviewed; get_history is stubbed to fail so
    # the review short-circuits to ERROR (no network), proving per-position error
    # isolation AND that dry-run never submits.
    positions = [
        SimpleNamespace(symbol="AVGO260702C00405000", qty="3", side="long",
                        avg_entry_price="14", market_value="4200", unrealized_pl="100",
                        asset_class="us_option"),
    ]
    broker = AlpacaBroker(client=_mock_client_with_positions(positions), dry_run=True)

    with patch.object(mp, "get_history", side_effect=HistoryError("no network")):
        result = manage_positions(broker, run_date=date(2026, 6, 9),
                                  forecaster_factory=lambda ctx: None)
    assert len(result.reviews) == 1
    r = result.reviews[0]
    assert r.action == "ERROR"
    assert r.underlying == "AVGO"   # OCC parse still succeeded before the failure
    assert r.order is None
    broker._client.submit_order.assert_not_called()


# ---------- exit-side open-orders awareness ----------

def _opt_position(symbol):
    return SimpleNamespace(symbol=symbol, qty="2", side="long", avg_entry_price="6",
                           market_value="900", unrealized_pl="-300", asset_class="us_option")


def test_manage_skips_resubmit_when_resting_sell_exists():
    sym = "META260618P00570000"
    client = MagicMock()
    client.get_all_positions.return_value = [_opt_position(sym)]
    client.get_orders.return_value = [
        SimpleNamespace(symbol=sym, side="sell", qty="2", limit_price="5.0",
                        status="new", id="o1"),
    ]
    broker = AlpacaBroker(dry_run=False, client=client)  # live; submit must NOT be called

    sell = PositionReview(symbol=sym, qty=2, action="SELL", bid=5.0, fair_value=4.2)
    with patch.object(mp, "review_position", return_value=sell):
        result = manage_positions(broker, run_date=date(2026, 6, 9),
                                  forecaster_factory=lambda ctx: None)
    r = result.reviews[0]
    assert r.action == "SELL"
    assert r.order is None
    assert "resting sell" in (r.reason or "")
    client.submit_order.assert_not_called()


def test_manage_submits_when_no_resting_sell():
    sym = "META260618P00570000"
    client = MagicMock()
    client.get_all_positions.return_value = [_opt_position(sym)]
    client.get_orders.return_value = []   # nothing resting
    broker = AlpacaBroker(dry_run=True, client=client)

    sell = PositionReview(symbol=sym, qty=2, action="SELL", bid=5.0, fair_value=4.2)
    with patch.object(mp, "review_position", return_value=sell):
        result = manage_positions(broker, run_date=date(2026, 6, 9),
                                  forecaster_factory=lambda ctx: None)
    r = result.reviews[0]
    assert r.order is not None
    assert r.order.status == ExecutionStatus.DRY_RUN
