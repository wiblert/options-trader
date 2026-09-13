"""
snapshot.py — freeze the S&P 500 universe + market caps to a dated CSV.

WHY A FROZEN SNAPSHOT
    The standardized test draws tickers with probability proportional to market
    cap. Market caps move every day, so if the sampler read live caps the drawn
    test set would silently change over time and the test would not be
    reproducible. Instead we snapshot the universe once, commit the dated CSV,
    and the sampler always reads from a frozen file. Re-snapshotting is a
    deliberate act (refresh_snapshot / the CLI --refresh-universe flag), and each
    snapshot is a new dated file — old ones are never overwritten.

UNIVERSE CHOICE
    S&P 500 constituents (~503 symbols, ~80% of total US equity market cap). This
    is the practical "overall stock market" proxy: assemblable reliably, and
    because cap-weighting concentrates probability mass on the largest names, the
    long tail of small-caps it omits carries negligible sampling weight anyway.

KNOWN BIASES (documented, not bugs)
    * Survivorship: the universe is today's listed constituents; firms that were
      delisted/acquired are absent. Acceptable for a forward forecasting-accuracy
      test, but it means the test set is not point-in-time historical.
    * Cap concentration: under power=1.0 cap-weighting, mega-caps dominate. This
      is the requested behaviour; the sampler exposes a `power` knob to flatten it.

CSV SCHEMA
    columns: ticker, market_cap, snapshot_date
    one row per constituent with a positive market cap; snapshot_date repeats the
    ISO date the snapshot was taken (also encoded in the filename).
"""

from __future__ import annotations

import io
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


logger = logging.getLogger(__name__)


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia 403s the default urllib User-Agent; a browser UA is required.
_BROWSER_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_HTTP_TIMEOUT_S = 30.0
SNAPSHOT_PREFIX = "sp500_"


def fetch_sp500_symbols() -> list[str]:
    """Return the current S&P 500 constituent symbols (yfinance-style, '.'→'-')."""
    html = requests.get(SP500_WIKI_URL, headers=_BROWSER_UA, timeout=_HTTP_TIMEOUT_S).text
    table = pd.read_html(io.StringIO(html))[0]
    symbols = table["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    out = sorted(set(symbols.tolist()))
    logger.info("Fetched %d S&P 500 symbols from Wikipedia", len(out))
    return out


def fetch_market_caps(symbols: list[str]) -> dict[str, float]:
    """Fetch market cap per symbol via yfinance fast_info.

    Symbols whose cap is missing/zero/non-finite are dropped (logged). Network
    errors per symbol are caught and the symbol is skipped — a snapshot of 490 of
    503 names is still a valid universe.
    """
    caps: dict[str, float] = {}
    failed: list[str] = []
    for s in symbols:
        try:
            mc = yf.Ticker(s).fast_info.market_cap
        except Exception as exc:  # noqa: BLE001 — yfinance raises a zoo of types
            failed.append(s)
            logger.debug("market cap fetch failed for %s: %r", s, exc)
            continue
        if mc is None or not (mc > 0) or not pd.notna(mc):
            failed.append(s)
            continue
        caps[s] = float(mc)
    if failed:
        logger.warning("No usable market cap for %d/%d symbols: %s",
                       len(failed), len(symbols), ", ".join(failed[:20]) + ("…" if len(failed) > 20 else ""))
    logger.info("Resolved market caps for %d symbols", len(caps))
    return caps


def refresh_snapshot(snapshot_date: Optional[date] = None) -> Path:
    """Fetch the live S&P 500 universe + caps and write a new dated snapshot CSV.

    Args:
        snapshot_date: date to stamp the snapshot with (defaults to today). Passed
            in by the CLI so the value is explicit and testable.

    Returns:
        Path to the written CSV (SNAPSHOT_DIR / sp500_<date>.csv).
    """
    if snapshot_date is None:
        snapshot_date = date.today()
    symbols = fetch_sp500_symbols()
    caps = fetch_market_caps(symbols)
    if not caps:
        raise RuntimeError("market cap fetch returned nothing; refusing to write empty snapshot")

    df = pd.DataFrame(
        {"ticker": list(caps.keys()), "market_cap": list(caps.values())}
    ).sort_values("market_cap", ascending=False).reset_index(drop=True)
    df["snapshot_date"] = snapshot_date.isoformat()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{SNAPSHOT_PREFIX}{snapshot_date.isoformat()}.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote snapshot %s (%d tickers)", path.name, len(df))
    return path


def latest_snapshot_path() -> Path:
    """Path to the most recent snapshot CSV (lexically max date in filename)."""
    if not SNAPSHOT_DIR.exists():
        raise FileNotFoundError(
            f"no snapshots dir at {SNAPSHOT_DIR}; run refresh_snapshot() first"
        )
    files = sorted(SNAPSHOT_DIR.glob(f"{SNAPSHOT_PREFIX}*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no snapshot CSVs in {SNAPSHOT_DIR}; run refresh_snapshot() first"
        )
    return files[-1]


def load_snapshot(path: Optional[Path] = None) -> pd.DataFrame:
    """Load a frozen snapshot CSV. Defaults to the latest committed snapshot.

    Returns a DataFrame with columns [ticker, market_cap, snapshot_date], with
    only finite positive market caps retained.
    """
    if path is None:
        path = latest_snapshot_path()
    df = pd.read_csv(path)
    missing = {"ticker", "market_cap"} - set(df.columns)
    if missing:
        raise ValueError(f"snapshot {path} missing columns: {missing}")
    df = df[(df["market_cap"] > 0) & df["market_cap"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"snapshot {path} has no positive market caps")
    return df
