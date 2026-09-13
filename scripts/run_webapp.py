#!/usr/bin/env python
"""
run_webapp.py — launch the forecast diagnostic web UI.

Thin wrapper over `options_trader.webapp.server` so the app can be started from
the repo root without remembering the module path:

    python scripts/run_webapp.py                 # binds 0.0.0.0:8000
    python scripts/run_webapp.py --port 9000 -v

Then open http://<vm-ip>:8000/ (open the port in the Azure NSG to reach it
from outside the VM). Forecasts use the same get_history / universe sampler the
trading pipeline uses, so ALPACA_API_KEY / ALPACA_SECRET_KEY should be set
(yfinance is the fallback).
"""

from options_trader.webapp.server import main


if __name__ == "__main__":
    main()
