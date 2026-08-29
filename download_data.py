"""Fetch the public Olist CSVs so the engine can be reproduced from a clean clone.

The dataset is the public Olist Brazilian E-Commerce release, mirrored on
HuggingFace. Only the five tables the engine actually reads are downloaded
(~59 MB); the 61 MB geolocation table is skipped because nothing uses it.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main"
FILES = [
    "olist_orders_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]
DEST = Path(__file__).resolve().parent / "data" / "raw"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  skip {name} (already present, {target.stat().st_size:,} bytes)")
            continue
        print(f"  fetching {name} ...", end="", flush=True)
        try:
            urllib.request.urlretrieve(f"{BASE}/{name}", target)
            print(f" {target.stat().st_size:,} bytes")
        except Exception as exc:
            print(f" FAILED: {exc}")
            return 1
    print(f"\nData ready in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
