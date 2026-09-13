"""Tests for the web UI's pure forecast payload builder (webapp/forecast_service.py).

No network: a synthetic StockReturnTS is fed directly to build_forecast_payload,
exactly as the server does after get_history. We assert the payload shape, that
it is JSON-serialisable, and that the diagnostic accuracy fields (PIT percentile,
log score, realized overlay) behave correctly under hold-out.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.webapp.forecast_service import build_forecast_payload


def _ts(n: int = 200, ticker: str = "TEST") -> StockReturnTS:
    rng = np.random.default_rng(0)
    rets = np.concatenate([[0.0], rng.normal(0.0003, 0.012, n - 1)])
    close = 100.0 * np.exp(np.cumsum(rets))
    # Give OHLC a little spread so the candlestick has real bars.
    high = close * 1.005
    low = close * 0.995
    dates = pd.bdate_range(end="2026-01-02", periods=n).to_numpy()
    return StockReturnTS(
        ticker=ticker, dates=dates, open=close, high=high, low=low, close=close,
        volume=np.full(n, 1e6), source="test", start=date(2024, 1, 1), end=date(2026, 1, 3),
    )


def test_payload_is_json_serialisable_and_shaped():
    p = build_forecast_payload(_ts(), horizon=21, mode="diagnostic", n_paths=500)
    # Round-trips through JSON (the server json.dumps it).
    json.dumps(p)

    assert p["ticker"] == "TEST"
    assert p["mode"] == "diagnostic"
    assert p["horizon"] == 21
    assert p["spot"] > 0

    # path_x = anchor + horizon future steps; bands align with it.
    assert len(p["path_x"]) == 22
    for key in ("p10", "p50", "p90"):
        assert len(p["bands"][key]) == 22

    # Spaghetti: default 30 paths, each anchored at spot and horizon+1 long.
    assert len(p["spaghetti"]) == 30
    assert all(len(path) == 22 for path in p["spaghetti"])
    assert all(abs(path[0] - p["spot"]) < 1e-9 for path in p["spaghetti"])

    # Terminal PDF/CDF grids align; CDF is monotone non-decreasing in [0, 1].
    g = p["terminal"]["grid"]
    for model in ("bootstrap", "gaussian"):
        assert len(p["terminal"][model]["pdf"]) == len(g)
        cdf = np.asarray(p["terminal"][model]["cdf"])
        assert len(cdf) == len(g)
        assert np.all(np.diff(cdf) >= -1e-9)
        assert cdf[0] >= -1e-9 and cdf[-1] <= 1.0 + 1e-9


def test_bands_ordered_and_widen_over_horizon():
    p = build_forecast_payload(_ts(), horizon=30, mode="diagnostic", n_paths=2000)
    p10 = np.asarray(p["bands"]["p10"])
    p50 = np.asarray(p["bands"]["p50"])
    p90 = np.asarray(p["bands"]["p90"])
    assert np.all(p10 <= p50) and np.all(p50 <= p90)
    # Step 0 is the anchor — zero spread; the band widens by the end.
    assert (p90[0] - p10[0]) < (p90[-1] - p10[-1])


def test_diagnostic_overlay_and_metrics():
    horizon = 21
    ts = _ts()
    p = build_forecast_payload(ts, horizon=horizon, mode="diagnostic", n_paths=2000)

    d = p["diagnostics"]
    assert d is not None
    # Realized terminal equals the true held-out close `horizon` steps after anchor.
    expected_terminal = float(ts.close[-1])
    assert d["realized_terminal"] == pytest.approx(expected_terminal)

    # Realized overlay: anchor + horizon points, starting at spot.
    r = p["realized"]
    assert len(r["close"]) == horizon + 1
    assert r["close"][0] == pytest.approx(p["spot"])
    assert r["close"][-1] == pytest.approx(expected_terminal)

    for model in ("bootstrap", "gaussian"):
        assert 0.0 <= d[model]["percentile"] <= 1.0
        assert np.isfinite(d[model]["log_score"])


def test_live_mode_has_no_realized():
    p = build_forecast_payload(_ts(), horizon=21, mode="live", n_paths=500)
    assert p["realized"] is None
    assert p["diagnostics"] is None
    # Live anchor is the final close.
    assert p["spot"] == pytest.approx(float(_ts().close[-1]))


def test_history_window_capped():
    p = build_forecast_payload(_ts(n=300), horizon=10, mode="live", history_bars=60)
    assert len(p["history"]["dates"]) == 60
    assert len(p["history"]["close"]) == 60


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        build_forecast_payload(_ts(), horizon=0, mode="live")
    with pytest.raises(ValueError):
        build_forecast_payload(_ts(), horizon=21, mode="bogus")
    # Not enough history for the requested hold-out.
    with pytest.raises(ValueError):
        build_forecast_payload(_ts(n=40), horizon=21, mode="diagnostic")


def test_seed_is_reproducible():
    a = build_forecast_payload(_ts(), horizon=21, mode="diagnostic", n_paths=500, seed=7)
    b = build_forecast_payload(_ts(), horizon=21, mode="diagnostic", n_paths=500, seed=7)
    assert a["spaghetti"] == b["spaghetti"]
    assert a["terminal"]["bootstrap"]["median"] == b["terminal"]["bootstrap"]["median"]
