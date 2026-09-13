"""
history.py — daily OHLCV bars for a single ticker.

Primary source: Alpaca StockHistoricalDataClient (adjusted bars).
Fallback: yfinance auto-adjusted bars when Alpaca returns transient errors or empty data.

Returned StockReturnTS is indexed by trading date (US/Eastern calendar).
Close is split- and dividend-adjusted.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

from options_trader.data.stock_return_ts import StockReturnTS


logger = logging.getLogger(__name__)


DEFAULT_LOOKBACK_YEARS = 3
TRADING_DAYS_PER_YEAR = 252
MIN_BARS_THRESHOLD = 100  # below this, treat Alpaca response as failed
ALPACA_TIMEOUT_S = 30.0   # hard cap on a single Alpaca request
YFINANCE_TIMEOUT_S = 30.0 # hard cap on a single yfinance request


class HistoryError(Exception):
    """Base for all history acquisition failures."""


class CredentialsError(HistoryError):
    """Missing or invalid Alpaca credentials. Caller should NOT retry or fall back."""


class DataUnavailableError(HistoryError):
    """No bars returned by any source for the requested window."""


def _default_window() -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=int(DEFAULT_LOOKBACK_YEARS * 365.25))
    return start, end


def _from_alpaca(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Pull adjusted daily bars from Alpaca. Raises CredentialsError if creds
    are missing. Network/data errors propagate as alpaca.common.exceptions.APIError.
    Hard-capped at ALPACA_TIMEOUT_S via a worker thread."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise CredentialsError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")

    client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
        end=datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
        adjustment=Adjustment.ALL,
    )
    # alpaca-py 0.43.2 does not expose a timeout on StockHistoricalDataClient.
    # Wrap in a worker thread so the caller is bounded; the inner thread may
    # outlive the timeout (Python cannot kill threads) but the daemon flag
    # prevents it from blocking interpreter shutdown.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="alpaca-bars") as ex:
        future = ex.submit(client.get_stock_bars, req)
        try:
            resp = future.result(timeout=ALPACA_TIMEOUT_S)
        except FutureTimeoutError as exc:
            raise TimeoutError(
                f"Alpaca get_stock_bars exceeded {ALPACA_TIMEOUT_S}s for {ticker}"
            ) from exc
    df = resp.df
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == ticker].drop(columns=["symbol"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern").dt.normalize().dt.tz_localize(None)
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    return df.sort_index()


def _from_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        timeout=YFINANCE_TIMEOUT_S,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.sort_index()


def get_history(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> StockReturnTS:
    """Daily OHLCV bars for `ticker`, adjusted for splits and dividends.

    Tries Alpaca first; falls back to yfinance only on transient failures
    (network errors, rate limits, empty responses). Credential errors propagate
    immediately — we never silently demote to yfinance when auth is the problem.

    Raises:
        CredentialsError: ALPACA env vars unset.
        DataUnavailableError: both sources returned no usable data.
    """
    if start is None:
        start, _ = _default_window()
    if end is None:
        _, end = _default_window()

    if start >= end:
        raise HistoryError(f"start ({start}) must be before end ({end})")

    df = pd.DataFrame()
    source = "alpaca"
    alpaca_err: Exception | None = None

    try:
        df = _from_alpaca(ticker, start, end)
    except CredentialsError:
        raise  # never silently fall back on auth failures
    except APIError as exc:
        alpaca_err = exc
        logger.warning("Alpaca APIError for %s [%s, %s): %s", ticker, start, end, exc)
    except (OSError, TimeoutError) as exc:
        alpaca_err = exc
        logger.warning("Alpaca network error for %s [%s, %s): %s", ticker, start, end, exc)

    if len(df) < MIN_BARS_THRESHOLD:
        if alpaca_err is None and not df.empty:
            logger.warning(
                "Alpaca returned only %d bars (<%d) for %s [%s, %s); trying yfinance",
                len(df), MIN_BARS_THRESHOLD, ticker, start, end,
            )
        logger.info("Falling back to yfinance for %s", ticker)
        try:
            df = _from_yfinance(ticker, start, end)
            source = "yfinance"
        except (OSError, TimeoutError, ValueError) as exc:
            raise DataUnavailableError(
                f"Both sources failed for {ticker} [{start}, {end}). "
                f"Alpaca: {alpaca_err!r}; yfinance: {exc!r}"
            ) from exc

    if df.empty:
        raise DataUnavailableError(
            f"No bars for {ticker} in [{start}, {end}). Alpaca error: {alpaca_err!r}"
        )

    logger.info("Loaded %d bars for %s from %s", len(df), ticker, source)
    return StockReturnTS.from_dataframe(df, ticker=ticker, source=source, start=start, end=end)


def get_latest_price(ticker: str) -> float | None:
    """Latest live trade price for `ticker`, or None if unavailable.

    Daily bars (`get_history`) only carry the last *completed* session's close,
    which intraday is yesterday's print — anchoring the forecast there biases
    every valuation by the day's move so far. This returns the most recent
    trade so the forecast can be re-anchored on the current price.

    Best-effort: any failure (missing creds, network, no data) returns None so
    the caller can fall back to the last close rather than abort the run. Hard-
    capped at ALPACA_TIMEOUT_S via a worker thread (mirrors `_from_alpaca`).
    """
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        logger.warning("get_latest_price: ALPACA creds unset; cannot fetch live price for %s", ticker)
        return None

    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="alpaca-trade") as ex:
            future = ex.submit(client.get_stock_latest_trade, req)
            resp = future.result(timeout=ALPACA_TIMEOUT_S)
        trade = resp.get(ticker) if resp else None
        price = float(trade.price) if trade is not None else 0.0
        if price > 0:
            return price
        logger.warning("get_latest_price: no usable trade for %s", ticker)
    except (APIError, OSError, FutureTimeoutError, KeyError, AttributeError, ValueError) as exc:
        logger.warning("get_latest_price: live price unavailable for %s: %s", ticker, exc)
    return None
