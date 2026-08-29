"""Local server for the live demo.

    python serve.py                 http://localhost:8000
    python serve.py --port 8080
    python serve.py --no-warm       skip preloading (first run will be slow)

Serves the dashboard from docs/ and adds two endpoints:

    GET /api/options          regions, metrics and the available window
    GET /api/run?...          runs an investigation, streaming each scope as it
                              fires (server-sent events), then the full result

Standard library only, so there is nothing to install before a demo. When the
dashboard is opened from GitHub Pages instead, /api/options is absent and the
page stays in static mode.

To share the running port, `ngrok http 8000` gives a public URL. That URL lets
anyone who has it trigger runs while it is open, so close the tunnel afterwards.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from engine import config as C
from engine.run import investigate, NpEncoder
from engine.generic_run import investigate_upload
from engine.ingest import read_csv_bytes, profile as build_profile

METRICS = ["review_score", "on_time", "delivery_days", "days_late", "days_to_carrier"]
WEB = C.ROOT / "docs"

# One investigation at a time. The engine shares a cached panel and a demo has a
# single operator; serialising keeps runs predictable rather than interleaved.
RUN_LOCK = threading.Lock()

# Uploaded files are held in memory only, keyed by a short id, and the oldest is
# dropped once a few are held. Nothing an operator uploads is written to disk.
UPLOADS: "dict[str, dict]" = {}
UPLOAD_KEEP = 4
MAX_UPLOAD = 60 * 1024 * 1024


def region_options():
    from engine import data as D
    p = D.load_panel()
    counts = p[p["delivered"]]["customer_state"].value_counts()
    return [{"code": s, "name": C.STATE_NAMES.get(s, s), "orders": int(n)}
            for s, n in counts.items()]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            print(f"  {self.address_string()}  {fmt % args}")

    # ---- helpers -------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj, cls=NpEncoder).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse(self, event, data):
        payload = json.dumps(data, cls=NpEncoder)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    # ---- routing -------------------------------------------------------
    def do_GET(self):
        route = urlparse(self.path)
        if route.path == "/api/options":
            return self.api_options()
        if route.path == "/api/run":
            return self.api_run(parse_qs(route.query))
        if route.path == "/api/run_upload":
            return self.api_run_upload(parse_qs(route.query))
        return super().do_GET()

    def do_POST(self):
        route = urlparse(self.path)
        if route.path == "/api/upload":
            return self.api_upload()
        self.send_error(404)

    # ---- upload -------------------------------------------------------
    def api_upload(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return self._json({"error": "no file received"}, 400)
        if n > MAX_UPLOAD:
            return self._json({"error": f"file exceeds "
                                        f"{MAX_UPLOAD // (1024*1024)} MB"}, 400)
        raw = self.rfile.read(n)
        name = (self.headers.get("X-Filename") or "upload.csv")[:120]
        try:
            df = read_csv_bytes(raw)
            prof = build_profile(df)
        except Exception as exc:
            return self._json({"error": str(exc)}, 400)

        uid = f"u{int(time.time()*1000) % 10**9}"
        UPLOADS[uid] = {"raw": raw, "name": name, "at": time.time()}
        for old in sorted(UPLOADS, key=lambda k: UPLOADS[k]["at"])[:-UPLOAD_KEEP]:
            UPLOADS.pop(old, None)
        print(f"  uploaded {name}  {len(raw):,} bytes  -> {uid}")
        self._json({"id": uid, "filename": name, "profile": prof.to_dict()})

    def api_run_upload(self, q):
        uid = (q.get("id") or [""])[0]
        metric = (q.get("metric") or [None])[0]
        item = UPLOADS.get(uid)
        if item is None:
            return self._json({"error": "upload not found; please pick the file again"}, 404)

        self._sse_open()
        if not RUN_LOCK.acquire(blocking=False):
            self._sse("error", {"message": "another run is in progress"})
            return
        lines: "queue.Queue[str|None]" = queue.Queue()
        box = {}

        def work():
            try:
                box["result"] = investigate_upload(
                    item["raw"], metric_key=metric, filename=item["name"],
                    on_log=lines.put)
            except Exception as exc:
                box["error"] = str(exc)
                traceback.print_exc()
            finally:
                lines.put(None)

        t0 = time.time()
        threading.Thread(target=work, daemon=True).start()
        try:
            while True:
                try:
                    line = lines.get(timeout=20)
                except queue.Empty:
                    self._sse("ping", {})
                    continue
                if line is None:
                    break
                self._sse("log", {"line": line})
            if "error" in box:
                self._sse("error", {"message": box["error"]})
            else:
                inv = box["result"]
                self._sse("done", {
                    "label": f"{item['name']} / {inv['meta']['metric_label']}",
                    "elapsed": round(time.time() - t0, 1),
                    "investigation": inv})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            RUN_LOCK.release()

    def api_options(self):
        try:
            self._json({
                "regions": region_options(),
                "metrics": METRICS,
                "window": {"start": C.PANEL_START, "end": C.PANEL_END},
            })
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def api_run(self, q):
        states = [s.upper() for s in q.get("state", []) if s.strip()]
        metric = (q.get("metric") or ["review_score"])[0]
        w_from = (q.get("from") or [None])[0] or None
        w_to = (q.get("to") or [None])[0] or None

        if metric not in METRICS:
            return self._json({"error": f"unknown metric: {metric}"}, 400)
        bad = [s for s in states if s not in C.STATE_NAMES]
        if bad:
            return self._json({"error": f"unknown region: {', '.join(bad)}"}, 400)

        self._sse_open()
        if not RUN_LOCK.acquire(blocking=False):
            self._sse("error", {"message": "another run is in progress"})
            return

        lines: "queue.Queue[str|None]" = queue.Queue()
        box = {}

        def work():
            try:
                box["result"] = investigate(
                    metric=metric, states=states or None,
                    week_from=w_from, week_to=w_to,
                    scenario="live", verbose=False, write=False,
                    on_log=lines.put)
            except Exception as exc:
                box["error"] = str(exc)
                traceback.print_exc()
            finally:
                lines.put(None)

        t0 = time.time()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    line = lines.get(timeout=20)
                except queue.Empty:
                    self._sse("ping", {})
                    continue
                if line is None:
                    break
                self._sse("log", {"line": line})
            worker.join(timeout=5)
            if "error" in box:
                self._sse("error", {"message": box["error"]})
            else:
                where = ", ".join(C.STATE_NAMES.get(s, s) for s in states) \
                    if states else "All regions"
                self._sse("done", {
                    "label": f"{where} / {metric.replace('_', ' ')}",
                    "elapsed": round(time.time() - t0, 1),
                    "investigation": box["result"],
                })
        except (BrokenPipeError, ConnectionResetError):
            pass                      # viewer navigated away mid-run
        finally:
            RUN_LOCK.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", "-p", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to accept connections from your network")
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args()

    if not a.no_warm:
        from engine import data as D
        t0 = time.time()
        print("Loading dataset ...", end=" ", flush=True)
        p = D.load_panel()
        print(f"{len(p):,} orders in {time.time() - t0:.1f}s")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"\n  Dashboard   http://localhost:{a.port}")
    print(f"  Serving     {WEB}")
    print(f"  Share with  ngrok http {a.port}")
    print("\n  Ctrl+C to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
