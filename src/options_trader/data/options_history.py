"""
options_history.py — historical End-of-Day option bars (for BACKTESTING).

The Option-Implied Beta forecaster needs the index (SPY/IWM) risk-neutral PDF on
each historical run date to backtest the theory. Live snapshots (options_chain.py)
only give TODAY's chain, so this module fetches historical EOD option aggregations
that can be replayed through the Breeden-Litzenberger derivation date-by-date.

    backtest:    options_history → EOD chain per date → Breeden-Litzenberger → PDF
    production:  options_chain (live snapshot) → Breeden-Litzenberger → PDF
                 (this module is bypassed live)

PROVIDER ABSTRACTION
    `OptionBarsProvider` is a Protocol so the data source is swappable: V1 uses
    `AlpacaOptionBarsProvider` (OptionHistoricalDataClient.get_option_bars with
    TimeFrame.Day). If Alpaca's option history rate limits / history depth prove
    insufficient, a `DatabentoOptionBarsProvider` can implement the same Protocol
    with no change to the forecaster or the BL layer.

Mirrors history.py / options_chain.py: missing creds → CredentialsError
(propagates, never silent); each network call is hard-capped by a worker-thread
timeout; failures surface as the typed OptionsHistoryUnavailableError. The pure
`_bars_from_alpaca_df` merge core is unit-testable with no network.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional, Protocol, Sequence, runtime_checkable

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from options_trader.data.history import CredentialsError


logger = logging.getLogger(__name__)


ALPACA_TIMEOUT_S = 30.0
MAX_SYMBOLS_PER_REQUEST = 100  # Alpaca caps the multi-symbol bar request size


class OptionsHistoryError(Exception):
    """Base for all historical-options-bar acquisition failures."""


class OptionsHistoryUnavailableError(OptionsHistoryError):
    """The provider returned no usable bars for the request."""


@dataclass(frozen=True)
class OptionBar:
    """One EOD aggregation for a single option contract."""

    symbol: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@runtime_checkable
class OptionBarsProvider(Protocol):
    """A swappable source of historical EOD option bars (Alpaca, Databento, …)."""

    def get_eod_bars(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> dict[str, list[OptionBar]]:
        """Return EOD bars keyed by contract symbol, oldest -> newest per symbol.

        A symbol with no bars in the window may be omitted from the dict.
        """
        ...


# ---------- pure parse core (no I/O — unit-testable) ----------

def _bars_from_alpaca_df(df: Optional[pd.DataFrame]) -> dict[str, list[OptionBar]]:
    """Parse an Alpaca option-bars DataFrame into OptionBars keyed by symbol.

    The alpaca-py `.df` is a MultiIndex (symbol, timestamp) frame with OHLCV
    columns. Robust to the single-index shape (one symbol) and an empty frame.
    """
    out: dict[str, list[OptionBar]] = {}
    if df is None or len(df) == 0:
        return out

    df = df.reset_index()
    if "symbol" not in df.columns or "timestamp" not in df.columns:
        raise OptionsHistoryError(
            f"unexpected option-bars frame columns: {list(df.columns)}"
        )

    ts = pd.to_datetime(df["timestamp"])
    # tz-aware -> US/Eastern calendar date; tz-naive taken as-is.
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("US/Eastern")
    df = df.assign(bardate=ts.dt.normalize().dt.date)

    for symbol, grp in df.groupby("symbol", sort=True):
        grp = grp.sort_values("timestamp")
        out[str(symbol)] = [
            OptionBar(
                symbol=str(symbol),
                bar_date=row.bardate,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0) or 0.0),
            )
            for row in grp.itertuples(index=False)
        ]
    return out


# ---------- Alpaca provider ----------

def _credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise CredentialsError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    return api_key, secret_key


def _with_timeout(fn, *args, label: str):
    """Run `fn(*args)` on a worker thread bounded by ALPACA_TIMEOUT_S (see history.py)."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="alpaca-optbars") as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=ALPACA_TIMEOUT_S)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"Alpaca {label} exceeded {ALPACA_TIMEOUT_S}s") from exc


class AlpacaOptionBarsProvider:
    """`OptionBarsProvider` backed by Alpaca's OptionHistoricalDataClient.

    `client` is injectable so tests can pass a stub. In production it is built
    lazily from the ALPACA env credentials on first use.
    """

    def __init__(self, client: Optional[OptionHistoricalDataClient] = None) -> None:
        self._client = client

    def _ensure_client(self) -> OptionHistoricalDataClient:
        if self._client is None:
            api_key, secret_key = _credentials()
            self._client = OptionHistoricalDataClient(api_key, secret_key)
        return self._client

    def get_eod_bars(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> dict[str, list[OptionBar]]:
        syms = list(symbols)
        if not syms:
            return {}
        if start > end:
            raise OptionsHistoryError(f"start ({start}) must be <= end ({end})")

        client = self._ensure_client()
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc)

        merged: dict[str, list[OptionBar]] = {}
        # Chunk the symbol list to stay within Alpaca's per-request cap.
        for i in range(0, len(syms), MAX_SYMBOLS_PER_REQUEST):
            chunk = syms[i:i + MAX_SYMBOLS_PER_REQUEST]
            req = OptionBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
            )
            try:
                resp = _with_timeout(client.get_option_bars, req, label="get_option_bars")
            except CredentialsError:
                raise
            except (APIError, OSError, TimeoutError) as exc:
                raise OptionsHistoryUnavailableError(
                    f"Failed to fetch option bars for {chunk[:3]}… ({len(chunk)} syms): {exc!r}"
                ) from exc
            merged.update(_bars_from_alpaca_df(getattr(resp, "df", None)))

        logger.info("Loaded EOD option bars for %d/%d requested symbols [%s, %s]",
                    len(merged), len(syms), start, end)
        return merged


# ---------- expired/active contract discovery (trading API) ----------

CONTRACTS_PAGE_LIMIT = 10_000
MAX_CONTRACT_PAGES = 20


@dataclass(frozen=True)
class ContractRef:
    """A discovered option contract's static terms (no quote)."""

    symbol: str
    strike: float
    expiry: date
    option_type: str  # "call" / "put"


def list_contracts(
    underlying: str,
    *,
    expiration_gte: date,
    expiration_lte: date,
    option_type: Optional[str] = None,
    status: str = "inactive",
    strike_gte: Optional[float] = None,
    strike_lte: Optional[float] = None,
    trading_client: Optional[TradingClient] = None,
) -> list[ContractRef]:
    """List option contracts for `underlying` in an expiration window.

    For BACKTESTING use `status="inactive"` — expired contracts are returned ONLY
    with the inactive status (active/none default `expiration_date_lte` to the next
    weekend, so a past expiry yields nothing). Paginated; `trading_client` injectable.
    """
    if status not in ("active", "inactive"):
        raise ValueError(f"status must be 'active' or 'inactive', got {status!r}")
    if trading_client is None:
        api_key, secret_key = _credentials()
        trading_client = TradingClient(api_key, secret_key, paper=True)

    ctype = ContractType(option_type) if option_type else None
    asset_status = AssetStatus.ACTIVE if status == "active" else AssetStatus.INACTIVE

    out: list[ContractRef] = []
    page_token = None
    for _ in range(MAX_CONTRACT_PAGES):
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=asset_status,
            type=ctype,
            expiration_date_gte=expiration_gte,
            expiration_date_lte=expiration_lte,
            strike_price_gte=str(strike_gte) if strike_gte is not None else None,
            strike_price_lte=str(strike_lte) if strike_lte is not None else None,
            limit=CONTRACTS_PAGE_LIMIT,
            page_token=page_token,
        )
        resp = _with_timeout(trading_client.get_option_contracts, req,
                             label="get_option_contracts")
        for c in (resp.option_contracts or []):
            ct = c.type.value if isinstance(c.type, ContractType) else str(c.type)
            out.append(ContractRef(
                symbol=c.symbol, strike=float(c.strike_price),
                expiry=c.expiration_date, option_type=ct.lower(),
            ))
        page_token = resp.next_page_token
        if not page_token:
            break
    return out


# ---------- convenience entry point ----------

def get_option_bars_eod(
    symbols: Sequence[str],
    start: date,
    end: date,
    provider: Optional[OptionBarsProvider] = None,
) -> dict[str, list[OptionBar]]:
    """Historical EOD option bars for `symbols` in [start, end].

    Args:
        symbols: OCC option contract symbols (e.g. "SPY260116C00500000").
        start, end: inclusive date window.
        provider: an OptionBarsProvider; defaults to AlpacaOptionBarsProvider.
            Swap in DatabentoOptionBarsProvider here if Alpaca limits bite.

    Raises:
        CredentialsError: ALPACA env vars unset (default provider only).
        OptionsHistoryUnavailableError: the provider failed.
    """
    if provider is None:
        provider = AlpacaOptionBarsProvider()
    return provider.get_eod_bars(symbols, start, end)
