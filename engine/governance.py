"""Source registry, KPI contract, method ledger and runtime telemetry.

Four things a reader is entitled to ask about a number on a dashboard:

    where did it come from      SOURCES     table, grain, refresh, freshness
    what does it mean           KPI_CONTRACT definition, formula, thresholds,
                                             drivers, owner, access
    how was it computed         METHODS      technique, and which class of
                                             method it belongs to
    what did it cost            Telemetry    latency, model calls, tokens, cost

The method ledger exists to answer one question directly: which parts of this
system are a language model, and which are not. In this engine no quantitative
value comes from a language model. The ledger says so and can be checked
against the code, rather than being a claim in a slide.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Sources: grain and refresh are properties of the feed, not of one query
# --------------------------------------------------------------------------

SOURCES = {
    "orders": {
        "table": "olist_orders_dataset",
        "grain": "one row per order",
        "refresh": "daily batch",
        "provides": ["order timestamps", "delivery dates", "order status"],
        "owner": "Fulfilment Data",
        "sensitivity": "internal",
    },
    "reviews": {
        "table": "olist_order_reviews_dataset",
        "grain": "one row per review (roughly one per order)",
        "refresh": "daily batch, arrives 1-3 days after delivery",
        "provides": ["review score", "free-text comment"],
        "owner": "Customer Experience",
        "sensitivity": "internal, free text may contain personal detail",
    },
    "order_items": {
        "table": "olist_order_items_dataset",
        "grain": "one row per item (many per order)",
        "refresh": "daily batch",
        "provides": ["price", "freight", "item count"],
        "owner": "Commerce",
        "sensitivity": "confidential (revenue)",
    },
    "customers": {
        "table": "olist_customers_dataset",
        "grain": "one row per customer",
        "refresh": "daily batch",
        "provides": ["state", "city"],
        "owner": "Commerce",
        "sensitivity": "internal",
    },
}

# The three grains above are the reason a KPI contract is needed at all: a
# review-grain metric and an item-grain metric cannot be averaged together
# without deciding, explicitly, what the denominator is.

KPI_CONTRACT = {
    "review_score": {
        "label": "Customer satisfaction",
        "definition": "Mean review score of delivered orders, by purchase week.",
        "formula": "mean(review_score) grouped by purchase week",
        "unit": "1-5 scale",
        "grain": "order",
        "sources": ["reviews", "orders"],
        "cadence": "weekly",
        "direction": "higher is better",
        "material_threshold": "robust z beyond -3, or a change past 0.15 points",
        "drivers": ["delivery timeliness", "product quality", "order accuracy"],
        "owner": "Customer Experience",
        "access": "internal",
    },
    "on_time": {
        "label": "On-time delivery rate",
        "definition": "Share of delivered orders arriving on or before the "
                      "estimated delivery date.",
        "formula": "mean(delivered_customer_date <= estimated_delivery_date)",
        "unit": "rate",
        "grain": "order",
        "sources": ["orders"],
        "cadence": "weekly",
        "direction": "higher is better",
        "material_threshold": "a change past 2 percentage points",
        "drivers": ["carrier capacity", "seller dispatch", "distance", "seasonality"],
        "owner": "Fulfilment",
        "access": "internal",
    },
    "delivery_days": {
        "label": "Delivery time",
        "definition": "Days from purchase to delivery at the customer.",
        "formula": "mean(delivered_customer_date - purchase_timestamp)",
        "unit": "days",
        "grain": "order",
        "sources": ["orders"],
        "cadence": "weekly",
        "direction": "lower is better",
        "material_threshold": "a change past 1 day",
        "drivers": ["carrier leg", "seller handoff", "distance"],
        "owner": "Fulfilment",
        "access": "internal",
    },
    "days_to_carrier": {
        "label": "Seller handoff time",
        "definition": "Days from purchase to the carrier collecting the parcel. "
                      "Separates seller-side delay from carrier-side delay.",
        "formula": "mean(delivered_carrier_date - purchase_timestamp)",
        "unit": "days",
        "grain": "order",
        "sources": ["orders"],
        "cadence": "weekly",
        "direction": "lower is better",
        "material_threshold": "a change past 0.5 days",
        "drivers": ["seller capacity", "stock availability"],
        "owner": "Seller Operations",
        "access": "internal",
    },
    "days_late": {
        "label": "Lateness",
        "definition": "Days between the estimated and actual delivery date. "
                      "Negative means early.",
        "formula": "mean(delivered_customer_date - estimated_delivery_date)",
        "unit": "days",
        "grain": "order",
        "sources": ["orders"],
        "cadence": "weekly",
        "direction": "lower is better",
        "material_threshold": "a change past 1 day",
        "drivers": ["carrier capacity", "estimate accuracy"],
        "owner": "Fulfilment",
        "access": "internal",
    },
    "revenue": {
        "label": "Revenue",
        "definition": "Sum of item prices for delivered orders, by purchase week.",
        "formula": "sum(price) grouped by purchase week",
        "unit": "BRL",
        "grain": "order item",
        "sources": ["order_items", "orders"],
        "cadence": "weekly",
        "direction": "higher is better",
        "material_threshold": "robust z beyond -3",
        "drivers": ["order volume", "average order value", "mix"],
        "owner": "Commerce",
        "access": "confidential",
    },
    "aov": {
        "label": "Average order value",
        "definition": "Mean item value per order, by purchase week.",
        "formula": "mean(order_value) grouped by purchase week",
        "unit": "BRL",
        "grain": "order",
        "sources": ["order_items", "orders"],
        "cadence": "weekly",
        "direction": "higher is better",
        "material_threshold": "robust z beyond -3",
        "drivers": ["basket size", "product mix", "discounting"],
        "owner": "Commerce",
        "access": "confidential",
    },
}

# --------------------------------------------------------------------------
# Method ledger
# --------------------------------------------------------------------------

DETERMINISTIC = "deterministic arithmetic"
STATISTICS = "classical statistics"
CAUSAL = "causal inference"
RETRIEVAL = "retrieval"
LLM = "language model"

METHODS = [
    {"scope": "Scope 0", "step": "Baseline and control limits",
     "technique": "Median / MAD robust baseline, +/-3 sigma control limits",
     "klass": STATISTICS,
     "why": "Median and MAD tolerate contamination, so a large anomaly cannot "
            "inflate the baseline it is measured against."},
    {"scope": "Scope 0", "step": "Change point",
     "technique": "Tabular CUSUM (k=0.5, h=4)", "klass": STATISTICS,
     "why": "Detects a sustained level shift rather than a single spike."},
    {"scope": "Scope 0", "step": "Significance of the movement",
     "technique": "Welch's t-test on order-level values", "klass": STATISTICS,
     "why": "Unequal variance and unequal sample sizes between the two windows."},
    {"scope": "Scope 1", "step": "Mix versus rate",
     "technique": "Additive decomposition of a weighted mean", "klass": DETERMINISTIC,
     "why": "Exact arithmetic. Separates a change in composition from a change "
            "in experience, with no model and no assumptions."},
    {"scope": "Scope 1", "step": "Segment scan",
     "technique": "Per-segment Welch t-test with Benjamini-Hochberg FDR control",
     "klass": STATISTICS,
     "why": "Scanning 27 segments at alpha 0.05 yields about 1.4 false positives "
            "by chance. FDR control stops noise being reported as insight."},
    {"scope": "Scope 2", "step": "Effect sizes and intervals",
     "technique": "Two-proportion z-test, Welch's t-test, percentile bootstrap",
     "klass": STATISTICS,
     "why": "Every reported effect carries an interval, so magnitude and "
            "uncertainty travel together."},
    {"scope": "Scope 2", "step": "Journey decomposition",
     "technique": "Leg-wise arithmetic on timestamp differences", "klass": DETERMINISTIC,
     "why": "Attributes delay to the carrier leg or the seller handoff, which "
            "names an owner rather than a symptom."},
    {"scope": "Scope 2", "step": "Customer language",
     "technique": "Lexicon match (built-in) or unsupervised term-frequency lift "
                  "with FDR control (uploads)",
     "klass": STATISTICS,
     "why": "For an unknown file no word list is supplied, so a finding cannot "
            "be an artefact of vocabulary chosen in advance."},
    {"scope": "Scope 2", "step": "External context",
     "technique": "Web search, returned as cited context", "klass": RETRIEVAL,
     "why": "Holidays, strikes and competitor moves are genuinely absent from "
            "the warehouse. Retrieved context is never scored as evidence."},
    {"scope": "Scope 3", "step": "Dose-response",
     "technique": "Linear regression of segment effect on segment exposure",
     "klass": CAUSAL,
     "why": "A real cause produces a gradient across segments; a coincidence "
            "does not."},
    {"scope": "Scope 3", "step": "Counterfactual",
     "technique": "Difference-in-differences with a parallel-trends test",
     "klass": CAUSAL,
     "why": "Compares exposed against unexposed segments, and tests the "
            "assumption the method depends on rather than assuming it."},
    {"scope": "Scope 3", "step": "Aggregation check",
     "technique": "Simpson's paradox test across segments", "klass": STATISTICS,
     "why": "An aggregate can move while every segment moves the other way."},
    {"scope": "Scope 4", "step": "Ranking and separability",
     "technique": "Weighted composite of four measured components, compared "
                  "across cause families",
     "klass": DETERMINISTIC,
     "why": "Weights order findings the data already produced. They cannot "
            "create one."},
    {"scope": "Scope 4", "step": "Narrative",
     "technique": "Template composition over computed values", "klass": DETERMINISTIC,
     "why": "Deliberate. Sentences are assembled from measured quantities, so "
            "no figure in the story can be invented by a generator."},
    {"scope": "Scope 4", "step": "Experiment design",
     "technique": "Power analysis for two-proportion and two-sample tests",
     "klass": STATISTICS,
     "why": "Turns a recommendation into a test with a sample size and a "
            "duration."},
]


def method_summary() -> dict:
    by = {}
    for m in METHODS:
        by.setdefault(m["klass"], []).append(f"{m['scope']}: {m['step']}")
    return {
        "methods": METHODS,
        "by_class": by,
        "llm_steps": [m for m in METHODS if m["klass"] == LLM],
        "llm_in_quantitative_path": False,
        "statement": (
            "No quantitative value in this report is produced by a language "
            "model. Detection, decomposition, ranking, causal testing and "
            "experiment sizing are classical statistics, causal inference and "
            "exact arithmetic. A language model is used nowhere in the numeric "
            "path; the narrative is composed from computed values by template. "
            "Retrieved web context is labelled as context and excluded from "
            "evidence scoring."),
    }


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------

@dataclass
class Telemetry:
    """Per-run cost and latency. Model counters are real, and they are zero."""
    started: float = field(default_factory=time.perf_counter)
    scopes: list = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_calls: int = 0
    _open: dict = field(default_factory=dict)

    def start(self, scope: str):
        self._open[scope] = time.perf_counter()

    def stop(self, scope: str, detail: str = ""):
        t0 = self._open.pop(scope, None)
        if t0 is None:
            return
        self.scopes.append({"scope": scope, "ms": round((time.perf_counter() - t0) * 1000, 1),
                            "detail": detail})

    def to_dict(self) -> dict:
        total = round((time.perf_counter() - self.started) * 1000, 1)
        return {
            "total_ms": total,
            "scopes": self.scopes,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost_usd": round(
                (self.prompt_tokens / 1e6) * 3.0 + (self.completion_tokens / 1e6) * 15.0, 6),
            "retrieval_calls": self.retrieval_calls,
            "cost_note": ("Priced at 3.00 USD per million input tokens and 15.00 per "
                          "million output tokens. With no model calls the cost of an "
                          "investigation is compute only, and does not grow with the "
                          "number of KPIs monitored."),
        }


def data_freshness(panel) -> dict:
    """Observed freshness of the extract, against the cadence each source declares."""
    last = panel["order_purchase_timestamp"].max() if "order_purchase_timestamp" in panel else None
    rows = []
    for key, s in SOURCES.items():
        rows.append({"source": key, "table": s["table"], "grain": s["grain"],
                     "refresh": s["refresh"], "owner": s["owner"],
                     "sensitivity": s["sensitivity"],
                     "latest_record": str(last)[:10] if last is not None else None})
    return {"sources": rows,
            "latest_record": str(last)[:10] if last is not None else None,
            "note": ("Refresh is the cadence each feed declares. Latest record is what "
                     "the extract actually contains, so a stale feed is visible rather "
                     "than assumed.")}
