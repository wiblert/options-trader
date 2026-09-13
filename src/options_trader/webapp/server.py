"""
server.py — stdlib HTTP server for the forecast diagnostic web UI.

Zero new dependencies (Python's `http.server`); the frontend pulls Plotly from a
CDN. This module owns the network edges the pure `forecast_service` deliberately
avoids: ticker sampling (`universe.sampler`) and history loading
(`data.history.get_history`).

Run it:
    python -m options_trader.webapp.server            # binds 0.0.0.0:8000
    python -m options_trader.webapp.server --port 9000 --host 127.0.0.1

Then open http://<vm-ip>:8000/ (ensure the port is allowed in the Azure NSG).

Routes:
    GET  /                     -> the single-page forecast-diagnostics UI
    GET  /review               -> the interactive daily-review UI (static/review.html)
    GET  /api/random_ticker    -> {"ticker": "..."} (cap-weighted sample)
    GET  /api/forecast?...      -> the forecast_service payload as JSON
                                  params: ticker, horizon, mode, n_paths,
                                          n_spaghetti, history_bars, seed
    POST /api/review/run       -> compute a review session (no orders); returns the
                                  session_id + reviewable sells/buys + budget
    POST /api/review/execute   -> execute one approved item (sell-to-close / buy);
                                  returns the order result + updated budget
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from options_trader.data.history import CredentialsError, get_history
from options_trader.execution.broker import AlpacaBroker
from options_trader.universe.sampler import sample_tickers
from options_trader.webapp.forecast_service import build_forecast_payload
from options_trader.webapp.review_service import (
    ReviewSession,
    build_review_session,
    execute_item,
)


logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# In-memory review sessions (single-user local app). Each holds the computed
# PositionReview/Candidate objects + the broker so an approve can execute exactly
# what was reviewed. Keyed by an opaque session_id handed to the client.
SESSIONS: dict[str, ReviewSession] = {}


class ForecastHandler(BaseHTTPRequestHandler):
    """Routes the three endpoints; everything else 404s."""

    # Quieter, structured logging instead of the default stderr spam.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    _STATIC_TYPES = {
        ".js": "application/javascript", ".css": "text/css",
        ".html": "text/html; charset=utf-8", ".json": "application/json",
        ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    }

    def _send_static(self, route: str) -> None:
        """Serve a file under static/ (e.g. the vendored Plotly bundle), with a
        path-traversal guard so only files inside STATIC_DIR are reachable."""
        rel = route[len("/static/"):]
        root = STATIC_DIR.resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            self.send_error(404, "Not found")
            return
        self._send_file(target, self._STATIC_TYPES.get(target.suffix.lower(),
                                                        "application/octet-stream"))

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        route = parsed.path
        params = parse_qs(parsed.query)

        try:
            if route == "/" or route == "/index.html":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif route == "/review" or route == "/review.html":
                self._send_file(STATIC_DIR / "review.html", "text/html; charset=utf-8")
            elif route.startswith("/static/"):
                self._send_static(route)
            elif route == "/api/random_ticker":
                self._handle_random_ticker()
            elif route == "/api/forecast":
                self._handle_forecast(params)
            else:
                self.send_error(404, "Not found")
        except _BadRequest as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 — surface any error to the UI
            logger.error("request failed: %s\n%s", exc, traceback.format_exc())
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    # ---------------------- POST (review workflow) ----------------------

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        route = urlparse(self.path).path
        try:
            body = self._read_json()
            if route == "/api/review/run":
                self._handle_review_run(body)
            elif route == "/api/review/execute":
                self._handle_review_execute(body)
            else:
                self.send_error(404, "Not found")
        except _BadRequest as exc:
            self._send_json({"error": str(exc)}, status=400)
        except CredentialsError as exc:
            self._send_json({"error": f"credentials: {exc}"}, status=400)
        except Exception as exc:  # noqa: BLE001 — surface any error to the UI
            logger.error("POST %s failed: %s\n%s", route, exc, traceback.format_exc())
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise _BadRequest(f"invalid JSON body: {exc}") from exc
        if not isinstance(obj, dict):
            raise _BadRequest("JSON body must be an object")
        return obj

    def _handle_review_run(self, body: dict) -> None:
        dry_run = bool(body.get("dry_run", True))
        use_sample = bool(body.get("use_sample", False))

        if use_sample:
            from options_trader.run_daily import (
                DEFAULT_N_TICKERS,
                DEFAULT_WATCHLIST_POWER,
                _daily_watchlist_seed,
            )
            from datetime import date

            n = int(body.get("n_tickers", DEFAULT_N_TICKERS))
            tickers = sample_tickers(
                n=n, seed=_daily_watchlist_seed(date.today()), power=DEFAULT_WATCHLIST_POWER
            )
        else:
            tickers = _parse_tickers(body.get("tickers", ""))
            if not tickers:
                raise _BadRequest("enter at least one ticker, or load the daily sample")

        # The server owns the network edges the service deliberately avoids.
        broker = AlpacaBroker(dry_run=dry_run)
        session = build_review_session(broker, tickers, skip_manage=False)
        session._broker = broker  # attach for later per-item execution

        session_id = secrets.token_hex(8)
        SESSIONS[session_id] = session
        payload = dict(session.payload)
        payload["session_id"] = session_id
        payload["tickers"] = tickers
        self._send_json(payload)

    def _handle_review_execute(self, body: dict) -> None:
        session_id = body.get("session_id")
        item_id = body.get("item_id")
        override = bool(body.get("override", False))
        if not session_id or not item_id:
            raise _BadRequest("session_id and item_id are required")
        session = SESSIONS.get(session_id)
        if session is None:
            raise _BadRequest("unknown or expired session_id; re-run the review")
        result = execute_item(session, item_id, override=override)
        self._send_json(result)

    def _handle_random_ticker(self) -> None:
        # Fresh draw each click: a random seed into the deterministic sampler.
        seed = secrets.randbelow(1_000_000)
        ticker = sample_tickers(1, seed=seed, power=0.5)[0]
        self._send_json({"ticker": ticker, "seed": seed})

    def _handle_forecast(self, params: dict) -> None:
        ticker = _one(params, "ticker")
        if not ticker:
            raise _BadRequest("ticker is required")
        ticker = ticker.upper()
        horizon = _int(params, "horizon", 21)
        mode = _one(params, "mode") or "diagnostic"
        n_paths = _int(params, "n_paths", 2000)
        n_spaghetti = _int(params, "n_spaghetti", 30)
        history_bars = _int(params, "history_bars", 120)
        seed = _int(params, "seed", 42)

        try:
            ts = get_history(ticker)
        except Exception as exc:  # noqa: BLE001 — translate to a clean 400 for the UI
            raise _BadRequest(f"could not load history for {ticker}: {exc}") from exc

        payload = build_forecast_payload(
            ts, horizon=horizon, mode=mode, n_paths=n_paths,
            n_spaghetti=n_spaghetti, history_bars=history_bars, seed=seed,
        )
        self._send_json(payload)


class _BadRequest(Exception):
    """Client error -> 400 with a readable message."""


def _parse_tickers(raw: str) -> list[str]:
    """Split a free-text ticker box on commas/whitespace into upper-case symbols."""
    if not raw:
        return []
    parts = raw.replace(",", " ").split()
    seen: list[str] = []
    for p in parts:
        sym = p.strip().upper()
        if sym and sym not in seen:
            seen.append(sym)
    return seen


def _one(params: dict, key: str) -> str | None:
    vals = params.get(key)
    return vals[0] if vals else None


def _int(params: dict, key: str, default: int) -> int:
    raw = _one(params, key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise _BadRequest(f"{key} must be an integer, got {raw!r}") from exc


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), ForecastHandler)
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    logger.info("Forecast UI serving on http://%s:%d/  (bound to %s)", shown, port, host)
    print(f"Forecast UI: http://{shown}:{port}/   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast diagnostic web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
