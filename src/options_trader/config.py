"""
config.py — canonical defaults for the live production run.

All knobs that appear in run_daily / manage_positions / review_service are
defined here so the weekly run is reproducible and overrides are discoverable.
Backtest-specific defaults (n_paths=10_000, webapp n_paths=2_000) live in
their own modules — they are intentionally different.
"""

# Monte Carlo paths for the live production run. Backtests use 10_000 (speed);
# the webapp uses 2_000 (responsiveness). Only change this if the live run is
# too slow or not smooth enough.
DEFAULT_N_PATHS: int = 50_000

# Annualised continuously-compounded risk-free rate used for option discounting.
DEFAULT_RISK_FREE_RATE: float = 0.04

# ¼-Kelly applied to the raw full-Kelly fraction before sizing.
DEFAULT_KELLY_FRACTION: float = 0.25

# Hard per-position cap: never deploy more than this fraction of bankroll on
# one name, regardless of Kelly output.
DEFAULT_MAX_FRACTION: float = 0.05

# Target days-to-expiry when selecting the option contract to trade.
DEFAULT_TARGET_DTE: int = 21
