"""
occ.py — parse an OCC option symbol into its components.

A held position comes back from the broker as just an OCC symbol string (e.g.
``AVGO260702C00405000``); to reprice it we need the underlying, expiry, type and
strike. The OCC 21-character format has a fixed-width tail:

    <root><yymmdd><C|P><strike*1000, 8 digits>

The root (underlying) is variable length (1–6 chars), so we slice from the RIGHT:
the last 15 characters are always 6 (date) + 1 (type) + 8 (strike). Everything
before that is the underlying.

    AVGO 260702 C 00405000
    ^root ^date  ^ ^strike(=405.000)

Pure / no I/O so it is unit-testable without a broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from options_trader.valuation.option_valuer import OptionType


# Fixed-width tail: yymmdd(6) + type(1) + strike(8) = 15 chars.
_TAIL = 15
_STRIKE_DIVISOR = 1000.0


@dataclass(frozen=True)
class OccSymbol:
    underlying: str
    expiry: date
    option_type: OptionType
    strike: float
    symbol: str


def parse_occ_symbol(symbol: str) -> OccSymbol:
    """Parse an OCC option symbol. Raises ValueError on a malformed symbol."""
    s = symbol.strip().upper()
    if len(s) <= _TAIL:
        raise ValueError(f"OCC symbol too short: {symbol!r}")

    underlying = s[:-_TAIL]
    if not underlying.isalpha():
        raise ValueError(f"bad underlying root in {symbol!r}: {underlying!r}")

    date_part = s[-_TAIL:-9]      # yymmdd
    cp = s[-9]                    # C or P
    strike_part = s[-8:]          # strike * 1000, zero-padded

    if cp not in ("C", "P"):
        raise ValueError(f"bad option type char in {symbol!r}: {cp!r}")
    if not (date_part.isdigit() and strike_part.isdigit()):
        raise ValueError(f"non-numeric date/strike in {symbol!r}")

    yy, mm, dd = int(date_part[:2]), int(date_part[2:4]), int(date_part[4:6])
    expiry = date(2000 + yy, mm, dd)   # OCC years are 2-digit; 20xx for our horizon
    strike = int(strike_part) / _STRIKE_DIVISOR
    option_type = OptionType.CALL if cp == "C" else OptionType.PUT

    return OccSymbol(
        underlying=underlying,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
        symbol=s,
    )
