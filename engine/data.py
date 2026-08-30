"""Load Olist and build the analysis panel.

On-time is a date comparison, delivery time is a date difference, satisfaction
is the review score, and topic flags are regex matches on review text.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from . import config as C


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


_PANEL_CACHE: pd.DataFrame | None = None


def load_panel(force: bool = False) -> pd.DataFrame:
    """Order-level panel: one row per delivered order, with outcome + text."""
    global _PANEL_CACHE
    if _PANEL_CACHE is not None and not force:
        return _PANEL_CACHE

    orders = pd.read_csv(
        C.RAW / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp", "order_approved_at",
                     "order_delivered_carrier_date", "order_delivered_customer_date",
                     "order_estimated_delivery_date"],
    )
    reviews = pd.read_csv(C.RAW / "olist_order_reviews_dataset.csv", encoding="utf-8",
                          parse_dates=["review_creation_date"])
    customers = pd.read_csv(C.RAW / "olist_customers_dataset.csv")
    items = pd.read_csv(C.RAW / "olist_order_items_dataset.csv")

    df = orders.merge(customers[["customer_id", "customer_state", "customer_city"]],
                      on="customer_id", how="left")

    # One review per order (a handful of orders have two; average the score and
    # concatenate the text rather than silently dropping one).
    rv = reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_text=("review_comment_message", lambda s: " ".join(x for x in s.dropna())),
        review_created=("review_creation_date", "min"),
    ).reset_index()
    df = df.merge(rv, on="order_id", how="left")

    # Basket value + freight, for the revenue KPI
    it = items.groupby("order_id").agg(
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
    ).reset_index()
    df = df.merge(it, on="order_id", how="left")

    # Average order value is the mean of per-order values, so the order-level
    # observation for `aov` is simply the order's own value. Naming it lets the
    # scopes test AOV the same way they test any other order-level metric.
    df["aov"] = df["order_value"]

    df["week"] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time
    df["delivered"] = df["order_status"].eq("delivered")

    # Cancellation is the one metric that must see orders which never shipped,
    # so it is measured over every order rather than the delivered subset. It
    # carries a value on every row; the delivery metrics stay null off that
    # subset and are dropped by the tests, which is why both can share a panel.
    df["cancellation_rate"] = df["order_status"].eq("canceled").astype(float)
    df["items_per_order"] = df["n_items"]

    # --- outcome metrics (all derived from dates, none assumed) ---
    df["on_time"] = np.where(
        df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna(),
        (df["order_delivered_customer_date"] <= df["order_estimated_delivery_date"]).astype(float),
        np.nan,
    )
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    df["days_late"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    # carrier handoff latency, separating seller-side from carrier-side delay
    df["days_to_carrier"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # --- unstructured: topic flags from the customer's own words ---
    df["review_text"] = df["review_text"].fillna("")
    df["has_text"] = df["review_text"].str.len().gt(0)
    norm = df["review_text"].map(_strip_accents)
    for topic, pattern in C.TOPIC_PATTERNS.items():
        df[f"topic_{topic}"] = norm.str.contains(pattern, regex=True, na=False).astype(float)

    df = df[(df["week"] >= C.PANEL_START) & (df["week"] <= C.PANEL_END)].copy()
    _PANEL_CACHE = df
    return df


def weekly(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Aggregate the order panel to weekly KPIs, optionally per segment."""
    keys = ["week"] + ([by] if by else [])
    d = df[df["delivered"]]
    # Cancellation is a property of the full order book, not of the delivered
    # subset, so its denominator is every order placed that week.
    cancels = df.groupby(keys).agg(
        cancellation_rate=("cancellation_rate", "mean"),
        orders_placed=("order_id", "size"),
    ).reset_index()
    agg = d.groupby(keys).agg(
        orders=("order_id", "size"),
        revenue=("order_value", "sum"),
        aov=("order_value", "mean"),
        on_time=("on_time", "mean"),
        delivery_days=("delivery_days", "mean"),
        days_late=("days_late", "mean"),
        days_to_carrier=("days_to_carrier", "mean"),
        review_score=("review_score", "mean"),
        items_per_order=("items_per_order", "mean"),
        n_reviews=("review_score", "count"),
    ).reset_index()
    agg = agg.merge(cancels, on=keys, how="outer")

    txt = d[d["has_text"]].groupby(keys).agg(
        n_text=("order_id", "size"),
        **{f"topic_{t}": (f"topic_{t}", "mean") for t in C.TOPIC_PATTERNS},
    ).reset_index()
    return agg.merge(txt, on=keys, how="left")


def topic_counts(df: pd.DataFrame, mask: pd.Series) -> dict[str, tuple[int, int]]:
    """(hits, total) per topic for a subset."""
    sub = df[mask & df["has_text"]]
    n = len(sub)
    return {t: (int(sub[f"topic_{t}"].sum()), n) for t in C.TOPIC_PATTERNS}
