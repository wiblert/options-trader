"""Tests for AlpacaBroker. The TradingClient is injected as a mock (no network)."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide
from alpaca.trading.requests import LimitOrderRequest

from options_trader.data.history import CredentialsError
from options_trader.execution import broker as broker_mod
from options_trader.execution.broker import (
    AlpacaBroker,
    ExecutionStatus,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.sizing.kelly import KellySizer
from options_trader.valuation.option_valuer import OptionContract, OptionType, OptionValuer


VAL_DATE = date(2026, 6, 2)


# ---------- helpers: build a real SIZED KellySizing through the chain ----------

def _sizing(p_up=0.6, ask=10.0, bankroll=130_000, symbol="META260618C00600000"):
    pd = PriceDistribution(np.array([120.0, 80.0]), weights=np.array([p_up, 1 - p_up]))
    contract = OptionContract(
        symbol=symbol, underlying="META", strike=100.0,
        expiry=date(2026, 6, 18), option_type=OptionType.CALL,
        bid=ask * 0.98, ask=ask,
    )
    val = OptionValuer(0.0).value(pd, contract, spot=100.0, valuation_date=VAL_DATE)
    sizing = KellySizer(kelly_fraction=0.25, max_fraction=0.5).size(pd, val, bankroll=bankroll)
    return pd, sizing


def _account(options_bp="130000", status="ACTIVE", blocked=False):
    return SimpleNamespace(
        equity="130000", cash="130000", buying_power="130000",
        options_buying_power=options_bp, options_trading_level="2",
        status=status, trading_blocked=blocked,
    )


def _mock_client(account=None, order_status="accepted"):
    c = MagicMock()
    c.get_account.return_value = account or _account()
    c.submit_order.return_value = SimpleNamespace(id="order-123", status=order_status)
    c.get_all_positions.return_value = []
    return c


# ---------- happy path ----------

def test_execute_submits_limit_buy():
    _, sizing = _sizing()
    client = _mock_client()
    broker = AlpacaBroker(client=client)
    res = broker.execute(sizing)

    assert res.status == ExecutionStatus.SUBMITTED
    assert res.order_id == "order-123"
    assert res.qty == sizing.n_contracts >= 1
    assert res.limit_price == 10.0
    # the submitted request is a LIMIT BUY for the contract symbol + qty
    req = client.submit_order.call_args.args[0] if client.submit_order.call_args.args \
        else client.submit_order.call_args.kwargs["order_data"]
    assert isinstance(req, LimitOrderRequest)
    assert req.symbol == sizing.valuation.contract.symbol
    assert req.side == OrderSide.BUY
    assert float(req.limit_price) == 10.0
    assert int(req.qty) == sizing.n_contracts


def test_limit_price_override():
    _, sizing = _sizing()
    client = _mock_client()
    res = AlpacaBroker(client=client).execute(sizing, limit_price=9.567)
    assert res.limit_price == 9.57  # rounded to the cent


# ---------- dry run ----------

def test_dry_run_does_not_submit():
    _, sizing = _sizing()
    client = _mock_client()
    res = AlpacaBroker(client=client, dry_run=True).execute(sizing)
    assert res.status == ExecutionStatus.DRY_RUN
    client.submit_order.assert_not_called()


# ---------- guards skip rather than submit ----------

def test_skip_when_not_sized():
    _, sizing = _sizing(p_up=0.5)  # fair bet → NO_EDGE, 0 contracts
    client = _mock_client()
    res = AlpacaBroker(client=client).execute(sizing)
    assert res.status == ExecutionStatus.SKIPPED
    client.submit_order.assert_not_called()
    client.get_account.assert_not_called()  # short-circuits before the account call


def test_skip_when_insufficient_buying_power():
    _, sizing = _sizing()  # cost = n * 10 * 100
    client = _mock_client(account=_account(options_bp="100"))
    res = AlpacaBroker(client=client).execute(sizing)
    assert res.status == ExecutionStatus.SKIPPED
    assert "buying power" in res.reason
    client.submit_order.assert_not_called()


def test_skip_when_trading_blocked():
    _, sizing = _sizing()
    client = _mock_client(account=_account(blocked=True))
    res = AlpacaBroker(client=client).execute(sizing)
    assert res.status == ExecutionStatus.SKIPPED
    assert "not tradable" in res.reason


def test_skip_when_account_not_active():
    _, sizing = _sizing()
    client = _mock_client(account=_account(status="ACCOUNT_UPDATED"))
    res = AlpacaBroker(client=client).execute(sizing)
    assert res.status == ExecutionStatus.SKIPPED


# ---------- rejection ----------

def test_apierror_returns_rejected():
    _, sizing = _sizing()
    client = _mock_client()
    client.submit_order.side_effect = APIError("contract not tradable")
    res = AlpacaBroker(client=client).execute(sizing)
    assert res.status == ExecutionStatus.REJECTED
    assert "not tradable" in res.reason


# ---------- account / positions mapping ----------

def test_account_snapshot_maps_fields():
    client = _mock_client()
    snap = AlpacaBroker(client=client).get_account_snapshot()
    assert snap.equity == 130000.0
    assert snap.options_buying_power == 130000.0
    assert snap.options_trading_level == 2
    assert snap.status == "ACTIVE"
    assert snap.trading_blocked is False


def test_get_positions_maps():
    client = _mock_client()
    client.get_all_positions.return_value = [
        SimpleNamespace(symbol="META260618C00600000", qty="3", side="long",
                        avg_entry_price="10.5", market_value="3300", unrealized_pl="150",
                        asset_class="us_option")
    ]
    pos = AlpacaBroker(client=client).get_positions()
    assert len(pos) == 1
    assert pos[0].symbol == "META260618C00600000"
    assert pos[0].qty == 3.0
    assert pos[0].asset_class == "us_option"


# ---------- safety invariants ----------

def test_missing_credentials_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(CredentialsError):
            AlpacaBroker()


def test_constructs_paper_client():
    with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}), \
         patch.object(broker_mod, "TradingClient") as TC:
        AlpacaBroker()
    TC.assert_called_once()
    assert TC.call_args.kwargs.get("paper") is True


def test_submit_buy_qty_zero_skips():
    client = _mock_client()
    res = AlpacaBroker(client=client).submit_buy("META260618C00600000", 0, 10.0)
    assert res.status == ExecutionStatus.SKIPPED
    client.submit_order.assert_not_called()


# ---------- sell-to-close path ----------

def test_submit_sell_submits_limit_sell():
    client = _mock_client()
    res = AlpacaBroker(client=client).submit_sell("AVGO260702C00405000", 3, 14.50)
    assert res.status == ExecutionStatus.SUBMITTED
    assert res.side == "sell"
    assert res.qty == 3
    assert res.limit_price == 14.50
    req = client.submit_order.call_args.args[0] if client.submit_order.call_args.args \
        else client.submit_order.call_args.kwargs["order_data"]
    assert isinstance(req, LimitOrderRequest)
    assert req.side == OrderSide.SELL
    assert req.symbol == "AVGO260702C00405000"
    assert int(req.qty) == 3


def test_submit_sell_dry_run_does_not_submit():
    client = _mock_client()
    res = AlpacaBroker(client=client, dry_run=True).submit_sell("AVGO260702C00405000", 3, 14.50)
    assert res.status == ExecutionStatus.DRY_RUN
    assert res.side == "sell"
    client.submit_order.assert_not_called()


def test_submit_sell_qty_zero_skips():
    client = _mock_client()
    res = AlpacaBroker(client=client).submit_sell("AVGO260702C00405000", 0, 14.50)
    assert res.status == ExecutionStatus.SKIPPED
    client.submit_order.assert_not_called()


def test_submit_sell_rejection_surfaces():
    client = _mock_client()
    client.submit_order.side_effect = APIError("not enough shares")
    res = AlpacaBroker(client=client).submit_sell("AVGO260702C00405000", 3, 14.50)
    assert res.status == ExecutionStatus.REJECTED
    assert "not enough" in (res.reason or "")


# ---------- open-orders read ----------

def test_get_open_orders_maps_fields():
    client = MagicMock()
    client.get_orders.return_value = [
        SimpleNamespace(symbol="AVGO260702C00405000", side=OrderSide.BUY, qty="3",
                        limit_price="14.74", status="new", id="o1"),
        SimpleNamespace(symbol="META260618P00570000", side=OrderSide.SELL, qty="2",
                        limit_price=None, status="new", id="o2"),
    ]
    orders = AlpacaBroker(client=client).get_open_orders()
    assert [(o.symbol, o.side, o.qty, o.limit_price) for o in orders] == [
        ("AVGO260702C00405000", "buy", 3.0, 14.74),
        ("META260618P00570000", "sell", 2.0, None),
    ]
