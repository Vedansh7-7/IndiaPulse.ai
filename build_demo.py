"""Build the demo bundle: one investigation per decision state.

The four scenarios below are not curated to flatter the engine. Each is a real
slice of the public dataset, and each lands on a different decision state
because the evidence in that slice genuinely differs. The INCONCLUSIVE and
NOISE cases are the point: an investigation engine that can only ever say
"here is your answer" is not trustworthy.
"""
from __future__ import annotations

import json
import shutil

from engine import config as C
from engine.run import investigate

SCENARIOS = [
    {"key": "national", "states": None,
     "title": "National - satisfaction decline",
     "blurb": "The full dataset. A sustained 11-week drop in customer satisfaction."},
    {"key": "parana", "states": ["PR"],
     "title": "Parana - ambiguous evidence",
     "blurb": "A real movement the evidence cannot attribute to a single cause."},
    {"key": "goias", "states": ["GO"],
     "title": "Goias - normal variation",
     "blurb": "A dip that looks alarming on a dashboard and is statistically ordinary."},
    {"key": "para", "states": ["PA"],
     "title": "Para - normal variation",
     "blurb": "A dip that does not persist. Detection needs a sustained run, "
              "not one extreme week."},
    {"key": "entitlement", "states": None, "metric": "aov",
     "title": "Average order value - restricted",
     "blurb": "A confidential KPI. The operations reader is told what is withheld, "
              "not shown a redacted number."},
    {"key": "delivery", "states": None, "metric": "delivery_days",
     "title": "Delivery time - second KPI",
     "blurb": "A different KPI over the same sources, at the same grain."},
]


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    # clear scratch files from exploratory sweeps
    for f in C.OUT.glob("tmp_*.json"):
        f.unlink()

    index = []
    for sc in SCENARIOS:
        print(f"\n{'=' * 70}\n{sc['key']}\n{'=' * 70}")
        r = investigate(states=sc["states"], metric=sc.get("metric", "review_score"),
                        scenario=sc["key"],
                        out_name=f"inv_{sc['key']}.json", verbose=True)
        d = r["scope4_decision"]
        index.append({
            "key": sc["key"], "title": sc["title"], "blurb": sc["blurb"],
            "file": f"inv_{sc['key']}.json",
            "state": d["state"], "headline": d["headline"],
            "orders": r["scope0_triage"].get("series") and sum(
                s["orders"] for s in r["scope0_triage"]["series"]),
        })

    (C.OUT / "index.json").write_text(
        json.dumps({"scenarios": index}, indent=2), encoding="utf-8")

    # mirror into docs/data so GitHub Pages can serve the site directly
    web_data = C.ROOT / "docs" / "data"
    web_data.mkdir(parents=True, exist_ok=True)
    for f in list(C.OUT.glob("inv_*.json")) + [C.OUT / "index.json"]:
        shutil.copy(f, web_data / f.name)

    # Also emit a plain JS bundle. fetch() is blocked on file:// origins, so
    # embedding the payload means a reviewer can open index.html directly with
    # no server and still see the full demo.
    bundle = {"index": index, "investigations": {}}
    for sc in SCENARIOS:
        bundle["investigations"][sc["key"]] = json.loads(
            (C.OUT / f"inv_{sc['key']}.json").read_text(encoding="utf-8"))
    (C.ROOT / "docs" / "data.js").write_text(
        "window.__INDIAPULSE__ = " + json.dumps(bundle) + ";",
        encoding="utf-8")

    print(f"\n{'=' * 70}\nDEMO BUNDLE")
    for i in index:
        print(f"  {i['state']:14s} {i['key']:16s} {i['title']}")
    print(f"\n-> {web_data}")


if __name__ == "__main__":
    main()
