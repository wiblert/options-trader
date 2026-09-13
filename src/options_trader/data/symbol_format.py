"""
symbol_format.py — ticker string translation between data-source conventions.

Our universe snapshot and yfinance both use dash for dual-class share tickers
(e.g. "BRK-B" — see `universe/snapshot.py`, which converts Wikipedia's dot
notation to dash specifically because yfinance expects dashes). Alpaca expects
the opposite: dot-separated class tickers ("BRK.B"). Only the outbound Alpaca
request needs translating — the canonical ticker used everywhere else in this
codebase (the snapshot, `StockReturnTS.ticker`, `OptionContract.underlying`)
stays dash-formatted, so this must be applied ONLY at the Alpaca call sites in
`history.py` / `options_chain.py`, never upstream of them.
"""

# Known dash/dot share-class tickers. Deliberately a small static table, not a
# generic dash->dot regex: US equities with a punctuation-bearing class suffix
# are rare and change rarely. Add an entry here if a new one surfaces (see
# docs/todo.md's "Symbol-format mapping for Alpaca" item).
_DASH_TO_ALPACA: dict[str, str] = {
    "BRK-B": "BRK.B",
    "BF-B": "BF.B",
}


def to_alpaca_symbol(ticker: str) -> str:
    """Translate a canonical (dash-format) ticker to what Alpaca expects.

    No-op for the vast majority of tickers; only rewrites the known
    dash-format share-class exceptions.
    """
    return _DASH_TO_ALPACA.get(ticker.upper(), ticker)
