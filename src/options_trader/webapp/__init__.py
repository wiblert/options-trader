"""
webapp — a local diagnostic web UI for inspecting forecasts.

`forecast_service` is a PURE, network-free core: it turns a loaded
`StockReturnTS` into a JSON-serialisable payload (history bars, bootstrap
spaghetti paths, terminal PDF/CDF, a Gaussian/normal comparison, and — in
diagnostic mode — the realized continuation plus PIT percentile and KDE log
score). `server` wires the network edges (get_history, ticker sampling) around
it and serves the static page.
"""
