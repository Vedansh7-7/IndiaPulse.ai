"""Run an investigation live, on parameters chosen at the moment.

    python demo.py --warm                  load the data before you present
    python demo.py --state RJ              investigate one region
    python demo.py --metric on_time        investigate a different KPI
    python demo.py --state SP RJ MG --from 2017-06-01 --to 2018-06-01
    python demo.py --list                  show what can be selected

Each run prints the scopes as they fire, then writes the result into the
dashboard as a new tab, so the same run can be shown as a log and as a page.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from engine import config as C
from engine.run import investigate, NpEncoder
from build_demo import SCENARIOS

METRICS = ["review_score", "on_time", "delivery_days", "days_late", "days_to_carrier"]

RULE = "-" * 68


def bar(title=""):
    print(f"\n{RULE}")
    if title:
        print(f" {title}")
        print(RULE)


def list_options():
    from engine import data as D
    p = D.load_panel()
    counts = p[p["delivered"]]["customer_state"].value_counts()
    bar("REGIONS  (code, orders, name)")
    rows = [(s, int(n), C.STATE_NAMES.get(s, s)) for s, n in counts.items()]
    for i in range(0, len(rows), 3):
        print("  " + "".join(f"{s:<4}{n:>7}  {nm:<22}" for s, n, nm in rows[i:i + 3]))
    bar("METRICS")
    for m in METRICS:
        print(f"  {m}")
    bar("WINDOW")
    print(f"  {C.PANEL_START}  to  {C.PANEL_END}")
    print()


def refresh_dashboard(live_payload, label):
    """Rebuild docs/data.js with the live run as the first tab."""
    index, investigations = [], {}
    index.append({
        "key": "live", "title": label, "blurb": "Run during the session.",
        "file": "inv_live.json", "state": live_payload["scope4_decision"]["state"],
        "headline": live_payload["scope4_decision"]["headline"],
    })
    investigations["live"] = live_payload
    for sc in SCENARIOS:
        f = C.OUT / f"inv_{sc['key']}.json"
        if not f.exists():
            continue
        inv = json.loads(f.read_text(encoding="utf-8"))
        index.append({
            "key": sc["key"], "title": sc["title"], "blurb": sc["blurb"],
            "file": f.name, "state": inv["scope4_decision"]["state"],
            "headline": inv["scope4_decision"]["headline"],
        })
        investigations[sc["key"]] = inv

    web = C.ROOT / "docs"
    (web / "data.js").write_text(
        "window.__INDIAPULSE__ = "
        + json.dumps({"index": index, "investigations": investigations}, cls=NpEncoder)
        + ";", encoding="utf-8")
    (web / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(C.OUT / "inv_live.json", web / "data" / "inv_live.json")
    return web / "index.html"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", "-s", nargs="+", metavar="CODE",
                    help="region code(s), e.g. RJ SP MG")
    ap.add_argument("--metric", "-m", default="review_score", choices=METRICS)
    ap.add_argument("--from", dest="week_from", metavar="YYYY-MM-DD")
    ap.add_argument("--to", dest="week_to", metavar="YYYY-MM-DD")
    ap.add_argument("--list", "-l", action="store_true", help="show selectable values")
    ap.add_argument("--warm", action="store_true",
                    help="preload the dataset so the next run is instant")
    ap.add_argument("--no-refresh", action="store_true",
                    help="do not update the dashboard")
    a = ap.parse_args()

    if a.list:
        list_options()
        return 0

    if a.warm:
        from engine import data as D
        t0 = time.time()
        print("Loading dataset ...", end=" ", flush=True)
        p = D.load_panel()
        print(f"{len(p):,} orders in {time.time() - t0:.1f}s. Ready.")
        return 0

    states = [s.upper() for s in a.state] if a.state else None
    if states:
        unknown = [s for s in states if s not in C.STATE_NAMES]
        if unknown:
            print(f"Unknown region code(s): {', '.join(unknown)}")
            print("Run  python demo.py --list  to see valid codes.")
            return 2

    where = ", ".join(C.STATE_NAMES.get(s, s) for s in states) if states else "All regions"
    window = ""
    if a.week_from or a.week_to:
        window = f"   {a.week_from or 'start'} to {a.week_to or 'end'}"
    label = f"{where} / {a.metric.replace('_', ' ')}"

    bar(f"LIVE RUN   {where}   metric: {a.metric}{window}")
    t0 = time.time()
    try:
        r = investigate(metric=a.metric, states=states,
                        week_from=a.week_from, week_to=a.week_to,
                        scenario="live", out_name="inv_live.json", verbose=True)
    except Exception as exc:
        print(f"\n  Could not complete: {exc}")
        print("  Try a larger region or a wider window; thin slices cannot be tested.")
        return 1
    elapsed = time.time() - t0

    d = r["scope4_decision"]
    t = r["scope0_triage"]
    bar(f"{d['state']}")
    for line in _wrap(d["headline"], 66):
        print(f"  {line}")
    print()
    print(f"  baseline {t['baseline_value']:.3f}    event {t['event_value']:.3f}"
          f"    change {t['delta']:+.3f}    robust z {t['robust_z']:.2f}")
    if d.get("separability", {}).get("note"):
        print()
        for line in _wrap(d["separability"]["note"], 66):
            print(f"  {line}")
    nt = d.get("next_test", {})
    if nt.get("recommendation"):
        print(f"\n  Next step: {nt['recommendation']}")
    print(f"\n  Completed in {elapsed:.1f}s")

    if not a.no_refresh:
        page = refresh_dashboard(r, label)
        print(f"  Dashboard updated. Reload {page} to see this run as the first tab.")
    print(RULE + "\n")
    return 0


def _wrap(text, width):
    words, line, out = str(text).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
