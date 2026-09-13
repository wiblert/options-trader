"""
price_chart.py — OHLC bar chart utility for a StockReturnTS.

Renders a traditional Western OHLC bar chart:
    * vertical line from low to high
    * left tick at open
    * right tick at close
    * up days green, down days red
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_trader.data.stock_return_ts import StockReturnTS


TICK_HALF_WIDTH_DAYS = 0.35   # how far the open/close ticks extend horizontally
UP_COLOR   = "#2ca02c"        # close >= open
DOWN_COLOR = "#d62728"        # close <  open


def plot_ohlc(ts: StockReturnTS, output_dir: str | Path = "output") -> Path:
    """Render an OHLC bar chart and save to <output_dir>/<ticker>_price.png.

    Returns the absolute Path of the saved PNG.
    """
    if len(ts) == 0:
        raise ValueError(f"cannot plot empty StockReturnTS for {ts.ticker}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ts.ticker}_price.png"

    dates = pd.to_datetime(ts.dates).to_pydatetime()
    x = mdates.date2num(dates)
    is_up = ts.close >= ts.open
    colors = np.where(is_up, UP_COLOR, DOWN_COLOR)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.vlines(x, ts.low, ts.high, colors=colors, linewidth=0.8)
    ax.hlines(ts.open,  x - TICK_HALF_WIDTH_DAYS, x, colors=colors, linewidth=0.8)
    ax.hlines(ts.close, x, x + TICK_HALF_WIDTH_DAYS, colors=colors, linewidth=0.8)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    d0 = pd.Timestamp(ts.dates[0]).date()
    d1 = pd.Timestamp(ts.dates[-1]).date()
    ax.set_title(f"{ts.ticker}  OHLC  {d0} → {d1}  ({len(ts)} bars, source={ts.source})")
    ax.set_ylabel("Price ($, adjusted)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.margins(x=0.005)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
