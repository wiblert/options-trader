"""
broker.py — Alpaca PAPER order placement for long option positions.

The final hop of the pipeline: takes a KellySizing decision and submits the order.

    forecast → value → size (KellySizing) → AlpacaBroker.execute → paper order

Safety stance (this is real-money-shaped code, kept deliberately conservative):
  * PAPER ONLY. The client is always constructed with paper=True; live trading is
    intentionally not reachable from this class.
  * LIMIT orders only. Option spreads are wide and a market order can fill far from
    the quote. The default limit is the contract ask (a marketable buy limit: fills
    at the ask or better, and preserves the edge the valuer computed against the ask).
    Pass a tighter limit (e.g. the mid) to be patient at the risk of no fill.
  * dry_run mode builds the order request but does not submit — for exercising the
    orchestrator end-to-end without placing anything.
  * Pre-submit account guards (active, not blocked, enough options buying power);
    a failed guard SKIPS the order with a reason rather than raising, so a daily
    batch keeps going. Only missing credentials raise.

BUY (long, `execute`/`submit_buy`) opens positions; SELL (`submit_sell`) closes them
(sell-to-close, used by manage_positions.py). We never open short positions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

from options_trader.data.history import CredentialsError
from options_trader.sizing.kelly import KellySizing, SizingStatus


logger = logging.getLogger(__name__)

DEFAULT_CONTRACT_MULTIPLIER = 100


class ExecutionStatus(str, Enum):
    SUBMITTED = "SUBMITTED"   # order accepted by Alpaca
    DRY_RUN = "DRY_RUN"       # built but not submitted (dry_run mode)
    SKIPPED = "SKIPPED"       # a pre-submit guard declined the order
    REJECTED = "REJECTED"     # Alpaca rejected the submission


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    options_buying_power: float
    options_trading_level: Optional[int]
    status: str
    trading_blocked: bool


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    asset_class: str


@dataclass(frozen=True)
class OrderSnapshot:
    """A currently-open (resting / unfilled) order — used for re-run dedup."""
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    limit_price: Optional[float]
    status: str
    order_id: str


@dataclass(frozen=True)
class OrderResult:
    status: ExecutionStatus
    symbol: str
    qty: int
    side: str
    limit_price: Optional[float]
    order_id: Optional[str] = None
    broker_status: Optional[str] = None   # Alpaca order.status
    reason: Optional[str] = None           # why skipped/rejected

    def __repr__(self) -> str:
        return (
            f"OrderResult({self.status.value} {self.side} {self.qty}x {self.symbol} "
            f"@{self.limit_price} id={self.order_id} reason={self.reason})"
        )


def _f(value, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _enum_str(value) -> str:
    """Stringify a value that may be a str-enum or a plain string."""
    return str(getattr(value, "value", value))


class AlpacaBroker:
    """Places long option orders against the Alpaca PAPER account."""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        client: Optional[TradingClient] = None,
        contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
    ) -> None:
        """
        Args:
            dry_run: build orders but never submit.
            client: inject a TradingClient (tests); when None, a PAPER client is
                built from ALPACA_API_KEY / ALPACA_SECRET_KEY.
            contract_multiplier: shares per contract (100) — for cost estimation.
        """
        self.dry_run = dry_run
        self.contract_multiplier = int(contract_multiplier)
        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            if not api_key or not secret_key:
                raise CredentialsError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
            self._client = TradingClient(api_key, secret_key, paper=True)  # PAPER ONLY

    # ---------- account / positions ----------

    def get_account_snapshot(self) -> AccountSnapshot:
        a = self._client.get_account()
        return AccountSnapshot(
            equity=_f(a.equity),
            cash=_f(a.cash),
            buying_power=_f(a.buying_power),
            options_buying_power=_f(getattr(a, "options_buying_power", None), default=_f(a.buying_power)),
            options_trading_level=(
                int(a.options_trading_level)
                if getattr(a, "options_trading_level", None) is not None else None
            ),
            status=_enum_str(a.status),
            trading_blocked=bool(a.trading_blocked),
        )

    def get_positions(self) -> list[PositionSnapshot]:
        return [
            PositionSnapshot(
                symbol=p.symbol,
                qty=_f(p.qty),
                side=_enum_str(p.side),
                avg_entry_price=_f(p.avg_entry_price),
                market_value=_f(p.market_value),
                unrealized_pl=_f(p.unrealized_pl),
                asset_class=_enum_str(p.asset_class),
            )
            for p in self._client.get_all_positions()
        ]

    def get_open_orders(self) -> list[OrderSnapshot]:
        """Currently-open (resting / unfilled) orders, both sides. Used by the
        daily passes to avoid double-submitting on a same-day re-run."""
        orders = self._client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        return [
            OrderSnapshot(
                symbol=o.symbol,
                side=_enum_str(o.side),
                qty=_f(o.qty),
                limit_price=(float(o.limit_price) if o.limit_price is not None else None),
                status=_enum_str(o.status),
                order_id=str(o.id),
            )
            for o in orders
        ]

    # ---------- order placement ----------

    def execute(self, sizing: KellySizing, *, limit_price: Optional[float] = None) -> OrderResult:
        """Submit a long BUY for a SIZED KellySizing, after account guards.

        Args:
            sizing: the sizer output. Anything other than status SIZED is skipped.
            limit_price: override the buy limit; defaults to the contract ask
                (marketable buy limit). Rounded to the cent.

        Returns:
            OrderResult. Guard failures return SKIPPED (not raised) so a batch can continue.
        """
        contract = sizing.valuation.contract
        symbol = contract.symbol
        qty = sizing.n_contracts

        def skip(reason: str) -> OrderResult:
            logger.info("Skipping %s: %s", symbol, reason)
            return OrderResult(ExecutionStatus.SKIPPED, symbol, qty, "buy", None, reason=reason)

        if sizing.status != SizingStatus.SIZED or qty < 1:
            return skip(f"sizing status {sizing.status.value}, n_contracts={qty}")

        if limit_price is None:
            limit_price = sizing.valuation.buy_price
        if limit_price is None or limit_price <= 0:
            return skip("no usable limit price (missing ask)")
        limit_price = round(float(limit_price), 2)

        acct = self.get_account_snapshot()
        if acct.trading_blocked or acct.status.lower() != "active":
            return skip(f"account not tradable (status={acct.status}, blocked={acct.trading_blocked})")
        est_cost = qty * limit_price * self.contract_multiplier
        if acct.options_buying_power < est_cost:
            return skip(
                f"insufficient options buying power: need ${est_cost:,.0f}, "
                f"have ${acct.options_buying_power:,.0f}"
            )

        return self.submit_buy(symbol, qty, limit_price)

    def submit_buy(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        *,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit a single limit BUY for `qty` contracts of `symbol`."""
        if qty < 1:
            return OrderResult(ExecutionStatus.SKIPPED, symbol, qty, "buy", limit_price,
                               reason="qty < 1")
        limit_price = round(float(limit_price), 2)
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

        if self.dry_run:
            logger.info("[dry_run] would BUY %d x %s @ %.2f", qty, symbol, limit_price)
            return OrderResult(ExecutionStatus.DRY_RUN, symbol, qty, "buy", limit_price,
                               reason="dry_run")

        try:
            order = self._client.submit_order(req)
        except APIError as exc:
            logger.warning("Order rejected for %s: %s", symbol, exc)
            return OrderResult(ExecutionStatus.REJECTED, symbol, qty, "buy", limit_price,
                               reason=str(exc))

        logger.info("Submitted BUY %d x %s @ %.2f (order %s, %s)",
                    qty, symbol, limit_price, order.id, _enum_str(order.status))
        return OrderResult(
            ExecutionStatus.SUBMITTED, symbol, qty, "buy", limit_price,
            order_id=str(order.id), broker_status=_enum_str(order.status),
        )

    def submit_sell(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        *,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit a single limit SELL for `qty` contracts of `symbol`.

        Used to CLOSE a long option position (sell-to-close). No buying-power guard
        — selling frees capital, not consumes it; Alpaca rejects a sell that exceeds
        the held quantity, surfaced here as REJECTED. The default limit is the
        contract bid (a marketable sell limit: fills at the bid or better).
        """
        if qty < 1:
            return OrderResult(ExecutionStatus.SKIPPED, symbol, qty, "sell", limit_price,
                               reason="qty < 1")
        limit_price = round(float(limit_price), 2)
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=time_in_force,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

        if self.dry_run:
            logger.info("[dry_run] would SELL %d x %s @ %.2f", qty, symbol, limit_price)
            return OrderResult(ExecutionStatus.DRY_RUN, symbol, qty, "sell", limit_price,
                               reason="dry_run")

        try:
            order = self._client.submit_order(req)
        except APIError as exc:
            logger.warning("Sell order rejected for %s: %s", symbol, exc)
            return OrderResult(ExecutionStatus.REJECTED, symbol, qty, "sell", limit_price,
                               reason=str(exc))

        logger.info("Submitted SELL %d x %s @ %.2f (order %s, %s)",
                    qty, symbol, limit_price, order.id, _enum_str(order.status))
        return OrderResult(
            ExecutionStatus.SUBMITTED, symbol, qty, "sell", limit_price,
            order_id=str(order.id), broker_status=_enum_str(order.status),
        )
