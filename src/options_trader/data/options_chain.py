"""
options_chain.py — tradable option contracts + live quotes from Alpaca.

The bridge between the forecast/valuation layers and a real order: it returns a
list of `OptionContract` (the valuation dataclass) for an underlying, populated
with the current market quote, so each one drops straight into `OptionValuer`.

Two Alpaca calls, merged on the contract symbol:
  1. TradingClient.get_option_contracts  — the MASTER LIST of tradable contracts
     (strike, expiry, type, open interest, tradable flag). Paginated.
  2. OptionHistoricalDataClient.get_option_chain — latest bid/ask + implied vol
     (+ greeks, which we ignore — our edge is a P-measure EV, not a greek model)
     keyed by symbol.

Only contracts that are TRADABLE are returned by default — we never hand the
sizer/executor a contract that can't fill. A tradable contract with no live quote
comes back with bid/ask=None (OptionValuer then reports NO_QUOTE), which is the
correct, honest state rather than a guess.

Mirrors history.py: missing creds → CredentialsError (propagates, never silent);
each network call is hard-capped by a worker-thread timeout; failures surface as
the typed OptionsDataUnavailableError. Alpaca-only (no fallback) for V1.

Feed note: paper / free-tier accounts cannot read the real-time OPRA feed, so the
default quote feed is "indicative". Pass feed="opra" if the account is entitled.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date
from typing import Optional

from alpaca.common.exceptions import APIError
from alpaca.data.enums import OptionsFeed
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from options_trader.data.history import CredentialsError
from options_trader.data.symbol_format import to_alpaca_symbol
from options_trader.valuation.option_valuer import OptionContract, OptionType


logger = logging.getLogger(__name__)


ALPACA_TIMEOUT_S = 30.0    # hard cap on a single Alpaca request
CONTRACTS_PAGE_LIMIT = 10_000  # max contracts per get_option_contracts page
MAX_PAGES = 20             # backstop against a runaway pagination loop


class OptionsChainError(Exception):
    """Base for all options-chain acquisition failures."""


class OptionsDataUnavailableError(OptionsChainError):
    """Alpaca returned no usable contracts for the request."""


# ---------- type coercion helpers ----------

def _to_option_type(raw) -> OptionType:
    """Map an Alpaca ContractType (enum or raw 'call'/'put' string) to OptionType."""
    value = raw.value if isinstance(raw, ContractType) else str(raw)
    return OptionType(value.lower())


def _opt_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _opt_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


# ---------- pure merge (no I/O — the unit-testable core) ----------

def _build_contracts(
    raw_contracts, snapshots: dict, include_untradable: bool, underlying: str
) -> list[OptionContract]:
    """Merge the tradable master list with the quote snapshots into OptionContracts.

    Args:
        raw_contracts: iterable of Alpaca trading OptionContract models.
        snapshots: dict symbol -> Alpaca OptionsSnapshot (may be missing symbols).
        include_untradable: keep contracts whose `tradable` flag is False.
        underlying: the CANONICAL (caller-supplied) underlying ticker, stamped
            onto every contract instead of Alpaca's own `rc.underlying_symbol` —
            for a dash/dot share-class ticker (e.g. "BRK-B") Alpaca reports its
            own dot-format symbol, and callers need contracts keyed by the same
            ticker string they asked for (see `symbol_format.py`).

    Returns:
        OptionContract list, sorted by (expiry, strike, type) for stable output.
    """
    out: list[OptionContract] = []
    for rc in raw_contracts:
        if not include_untradable and not getattr(rc, "tradable", True):
            continue

        snap = snapshots.get(rc.symbol)
        bid = ask = iv = None
        if snap is not None:
            quote = getattr(snap, "latest_quote", None)
            if quote is not None:
                bid = _opt_float(getattr(quote, "bid_price", None))
                ask = _opt_float(getattr(quote, "ask_price", None))
            iv = _opt_float(getattr(snap, "implied_volatility", None))

        out.append(
            OptionContract(
                symbol=rc.symbol,
                underlying=underlying,
                strike=float(rc.strike_price),
                expiry=rc.expiration_date,
                option_type=_to_option_type(rc.type),
                bid=bid,
                ask=ask,
                implied_vol=iv,
                open_interest=_opt_int(getattr(rc, "open_interest", None)),
            )
        )

    out.sort(key=lambda c: (c.expiry, c.strike, c.option_type.value))
    return out


# ---------- I/O: credentials, timeout, fetch ----------

def _credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise CredentialsError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    return api_key, secret_key


def _with_timeout(fn, *args, label: str):
    """Run `fn(*args)` on a worker thread, bounded by ALPACA_TIMEOUT_S.

    alpaca-py 0.43.2 exposes no per-request timeout, so we wrap as in history.py.
    The inner thread may outlive the timeout (Python can't kill threads) but the
    daemon executor prevents it blocking interpreter shutdown.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="alpaca-opt") as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=ALPACA_TIMEOUT_S)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"Alpaca {label} exceeded {ALPACA_TIMEOUT_S}s") from exc


def _fetch_contracts(
    underlying: str,
    option_type: Optional[OptionType],
    expiration_gte: Optional[date],
    expiration_lte: Optional[date],
    strike_gte: Optional[float],
    strike_lte: Optional[float],
) -> list:
    """Page through the tradable contract master list. Returns trading models."""
    api_key, secret_key = _credentials()
    client = TradingClient(api_key, secret_key, paper=True)

    contracts: list = []
    page_token = None
    for _ in range(MAX_PAGES):
        req = GetOptionContractsRequest(
            underlying_symbols=[to_alpaca_symbol(underlying)],
            status=AssetStatus.ACTIVE,
            type=ContractType(option_type.value) if option_type else None,
            expiration_date_gte=expiration_gte,
            expiration_date_lte=expiration_lte,
            strike_price_gte=str(strike_gte) if strike_gte is not None else None,
            strike_price_lte=str(strike_lte) if strike_lte is not None else None,
            limit=CONTRACTS_PAGE_LIMIT,
            page_token=page_token,
        )
        resp = _with_timeout(client.get_option_contracts, req, label="get_option_contracts")
        contracts.extend(resp.option_contracts or [])
        page_token = resp.next_page_token
        if not page_token:
            break
    else:
        logger.warning("get_option_contracts hit MAX_PAGES=%d for %s; results truncated",
                       MAX_PAGES, underlying)
    return contracts


def _fetch_snapshots(
    underlying: str,
    option_type: Optional[OptionType],
    expiration_gte: Optional[date],
    expiration_lte: Optional[date],
    strike_gte: Optional[float],
    strike_lte: Optional[float],
    feed: OptionsFeed,
) -> dict:
    """Fetch latest quote/IV/greek snapshots keyed by contract symbol."""
    api_key, secret_key = _credentials()
    client = OptionHistoricalDataClient(api_key, secret_key)
    req = OptionChainRequest(
        underlying_symbol=to_alpaca_symbol(underlying),
        feed=feed,
        type=ContractType(option_type.value) if option_type else None,
        expiration_date_gte=expiration_gte,
        expiration_date_lte=expiration_lte,
        strike_price_gte=strike_gte,
        strike_price_lte=strike_lte,
    )
    return _with_timeout(client.get_option_chain, req, label="get_option_chain") or {}


# ---------- public entry point ----------

def get_option_chain(
    underlying: str,
    *,
    option_type: Optional[OptionType] = None,
    expiration_gte: Optional[date] = None,
    expiration_lte: Optional[date] = None,
    strike_gte: Optional[float] = None,
    strike_lte: Optional[float] = None,
    feed: str = "indicative",
    include_untradable: bool = False,
) -> list[OptionContract]:
    """Tradable option contracts for `underlying`, populated with live quotes.

    All filters are optional but you almost always want at least an expiration
    window — an unfiltered chain on a liquid name is thousands of contracts.

    Args:
        underlying: e.g. "META".
        option_type: restrict to CALL or PUT; None returns both.
        expiration_gte / expiration_lte: expiry date window (inclusive).
        strike_gte / strike_lte: strike window (inclusive).
        feed: "indicative" (free/paper default) or "opra" (requires entitlement).
        include_untradable: also return contracts flagged non-tradable.

    Returns:
        OptionContract list sorted by (expiry, strike, type). Contracts with no
        live quote have bid/ask=None.

    Raises:
        CredentialsError: ALPACA env vars unset (propagates, no fallback).
        OptionsDataUnavailableError: no tradable contracts for the request.
    """
    try:
        feed_enum = OptionsFeed(feed)
    except ValueError as exc:
        raise OptionsChainError(
            f"feed must be one of {[f.value for f in OptionsFeed]}, got {feed!r}"
        ) from exc

    try:
        raw_contracts = _fetch_contracts(
            underlying, option_type, expiration_gte, expiration_lte, strike_gte, strike_lte
        )
    except CredentialsError:
        raise  # never swallow auth failures
    except (APIError, OSError, TimeoutError) as exc:
        raise OptionsDataUnavailableError(
            f"Failed to fetch option contracts for {underlying}: {exc!r}"
        ) from exc

    if not raw_contracts:
        raise OptionsDataUnavailableError(
            f"No tradable option contracts for {underlying} in the requested window"
        )

    # Quotes are best-effort: if the snapshot call fails we still return the
    # tradable contracts (quote-less). The valuer reports NO_QUOTE for those.
    snapshots: dict = {}
    try:
        snapshots = _fetch_snapshots(
            underlying, option_type, expiration_gte, expiration_lte,
            strike_gte, strike_lte, feed_enum,
        )
    except (APIError, OSError, TimeoutError) as exc:
        logger.warning("Quote snapshots unavailable for %s (%r); returning quote-less contracts",
                       underlying, exc)

    contracts = _build_contracts(raw_contracts, snapshots, include_untradable, underlying)
    logger.info("Loaded %d option contracts for %s (%d with quotes)",
                len(contracts), underlying, sum(c.bid is not None for c in contracts))
    return contracts
