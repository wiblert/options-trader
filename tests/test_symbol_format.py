"""Tests for the Alpaca symbol-format translation table."""

from options_trader.data.symbol_format import to_alpaca_symbol


def test_translates_known_dash_class_tickers():
    assert to_alpaca_symbol("BRK-B") == "BRK.B"
    assert to_alpaca_symbol("BF-B") == "BF.B"


def test_is_case_insensitive_on_input():
    assert to_alpaca_symbol("brk-b") == "BRK.B"


def test_leaves_normal_tickers_unchanged():
    assert to_alpaca_symbol("AAPL") == "AAPL"
    assert to_alpaca_symbol("GOOGL") == "GOOGL"
    assert to_alpaca_symbol("BRK-A") == "BRK-A"  # not a known mapping, passed through
