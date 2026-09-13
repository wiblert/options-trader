"""Tests for the OCC option-symbol parser (pure, no network)."""

from datetime import date

import pytest

from options_trader.data.occ import parse_occ_symbol
from options_trader.valuation.option_valuer import OptionType


def test_parses_call():
    occ = parse_occ_symbol("AVGO260702C00405000")
    assert occ.underlying == "AVGO"
    assert occ.expiry == date(2026, 7, 2)
    assert occ.option_type == OptionType.CALL
    assert occ.strike == 405.0


def test_parses_put():
    occ = parse_occ_symbol("META260618P00570000")
    assert occ.underlying == "META"
    assert occ.expiry == date(2026, 6, 18)
    assert occ.option_type == OptionType.PUT
    assert occ.strike == 570.0


def test_fractional_strike():
    # 00100500 -> 100.5
    occ = parse_occ_symbol("CCI260618C00100500")
    assert occ.strike == 100.5


def test_short_underlying():
    occ = parse_occ_symbol("F260618C00012000")
    assert occ.underlying == "F"
    assert occ.strike == 12.0


def test_case_insensitive():
    occ = parse_occ_symbol("avgo260702c00405000")
    assert occ.underlying == "AVGO"
    assert occ.option_type == OptionType.CALL


@pytest.mark.parametrize("bad", [
    "",                       # empty
    "AAPL",                   # no tail
    "AAPL260702X00405000",    # bad type char
    "AAPL2607020405000C0",    # type not at the right slot
    "1234260702C00405000",    # non-alpha underlying
])
def test_malformed_raises(bad):
    with pytest.raises(ValueError):
        parse_occ_symbol(bad)
