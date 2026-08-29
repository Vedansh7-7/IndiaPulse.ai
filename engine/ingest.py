"""Read an unfamiliar CSV and work out what is in it.

Nothing here is configured by the user. Each column is examined and assigned a
role from its own contents:

    date        parses as timestamps across most rows
    measure     numeric, and varies
    segment     text with few distinct values relative to row count
    identifier  text with nearly one distinct value per row
    text        free text: long values, mostly multi-word

From those roles a weekly panel is built and a set of KPIs is derived, so a
plain transaction export becomes something the scopes can investigate.

The profile is returned alongside the data. A reader can see what was inferred
and disagree with it, which matters more than the inference being clever.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

MAX_ROWS = 400_000
MAX_BYTES = 60 * 1024 * 1024

# Name hints only break ties. Role assignment is decided by content first, so a
# column called "amount" holding text is not treated as money.
MONEY_HINTS = re.compile(
    r"amount|revenue|sales|price|value|total|cost|spend|gmv|turnover|net|gross",
    re.I)
QTY_HINTS = re.compile(r"qty|quantity|units|count|items|volume", re.I)
DATE_HINTS = re.compile(r"date|time|day|week|month|created|order.*at|timestamp", re.I)
ID_HINTS = re.compile(r"\bid\b|_id|uuid|guid|code|ref|number|sku", re.I)


@dataclass
class Column:
    name: str
    role: str
    dtype: str
    n_unique: int
    null_rate: float
    confidence: float
    note: str = ""
    sample: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Profile:
    rows: int
    columns: list
    date_column: str | None
    segment_columns: list
    measure_columns: list
    text_column: str | None
    id_column: str | None
    money_column: str | None
    quantity_column: str | None
    metrics: list
    span: dict
    warnings: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d["columns"] = [c.to_dict() if isinstance(c, Column) else c for c in self.columns]
        return d


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Decode and parse a CSV without being told its encoding or separator."""
    if len(raw) > MAX_BYTES:
        raise ValueError(f"file is larger than {MAX_BYTES // (1024*1024)} MB")
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("could not decode the file as text")

    sample = text[:16000]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
        sep = max(counts, key=counts.get) if max(counts.values()) else ","

    df = pd.read_csv(io.StringIO(text), sep=sep, nrows=MAX_ROWS,
                     skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    if df.empty:
        raise ValueError("the file has no rows")
    if len(df.columns) < 2:
        raise ValueError("the file needs at least two columns")
    return df


# --------------------------------------------------------------------------
# role inference
# --------------------------------------------------------------------------

def _try_dates(s: pd.Series) -> tuple[pd.Series | None, float]:
    """Parse a column as dates and report the share that parsed."""
    sample = s.dropna()
    if sample.empty:
        return None, 0.0
    if pd.api.types.is_numeric_dtype(sample):
        # only treat numbers as dates when they look like epoch seconds/ms
        v = pd.to_numeric(sample, errors="coerce").dropna()
        if v.empty or not (1e8 < abs(v.median()) < 1e13):
            return None, 0.0
    fmts = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y",
            "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "ISO8601"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in fmts:
            try:
                parsed = pd.to_datetime(s, errors="coerce", format=f)
            except Exception:
                continue
            rate = float(parsed.notna().mean())
            if rate >= 0.80:
                return parsed, rate
        for kw in ({"format": "mixed"}, {"dayfirst": True}):
            try:
                parsed = pd.to_datetime(s, errors="coerce", **kw)
            except Exception:
                continue
            rate = float(parsed.notna().mean())
            if rate >= 0.80:
                return parsed, rate
    return None, 0.0


def classify(df: pd.DataFrame) -> list[Column]:
    n = len(df)
    out = []
    for name in df.columns:
        s = df[name]
        nulls = float(s.isna().mean())
        nun = int(s.nunique(dropna=True))
        ratio = nun / max(n, 1)
        role, conf, note = "unused", 0.0, ""

        parsed, rate = _try_dates(s)
        if parsed is not None and nun > 1:
            role, conf = "date", rate
            note = f"{rate:.0%} of values parse as dates"
        elif pd.api.types.is_numeric_dtype(s) and nun > 2:
            role, conf = "measure", 0.9
            note = "numeric and varying"
            if ID_HINTS.search(name) and ratio > 0.9:
                role, conf = "identifier", 0.8
                note = "numeric but nearly unique per row"
        else:
            vals = s.dropna().astype(str)
            avg_len = float(vals.str.len().mean()) if len(vals) else 0.0
            words = float(vals.str.count(r"\s+").mean() + 1) if len(vals) else 0.0
            # Sentence-shaped: several words and reasonable length. A raw
            # character threshold alone misclassifies short review text, which
            # is exactly the kind this engine reads.
            if words >= 3 and avg_len >= 16:
                role, conf = "text", min(1.0, (avg_len / 60) + (words / 20))
                note = f"free text, {avg_len:.0f} chars / {words:.1f} words on average"
            elif ratio > 0.85 and nun > 20:
                role, conf = "identifier", 0.85
                note = "nearly one distinct value per row"
            elif 1 < nun <= max(60, n * 0.05):
                role, conf = "segment", 1 - min(ratio, 0.9)
                note = f"{nun} distinct values"
            elif nun <= 1:
                role, conf, note = "unused", 1.0, "single value"
            else:
                role, conf = "identifier", 0.5
                note = f"{nun} distinct values, too many to group by"

        out.append(Column(
            name=name, role=role, dtype=str(s.dtype), n_unique=nun,
            null_rate=round(nulls, 4), confidence=round(float(conf), 3), note=note,
            sample=[str(v)[:60] for v in s.dropna().head(3).tolist()],
        ))
    return out


def profile(df: pd.DataFrame) -> Profile:
    cols = classify(df)
    warn = []
    by = lambda r: [c for c in cols if c.role == r]

    dates = sorted(by("date"), key=lambda c: (-c.confidence, -c.n_unique))
    if not dates:
        raise ValueError(
            "no date column found. The engine compares periods, so it needs one "
            "column of dates or timestamps.")
    date_col = dates[0].name
    if len(dates) > 1:
        warn.append(f"several date columns found; using '{date_col}'.")

    segs = sorted(by("segment"), key=lambda c: -c.confidence)
    measures = by("measure")
    texts = sorted(by("text"), key=lambda c: -c.confidence)
    ids = by("identifier")

    money = next((c.name for c in measures if MONEY_HINTS.search(c.name)), None)
    if money is None and measures:
        # fall back to the widest-spread positive measure, which is how a money
        # column usually behaves next to counts and ratings
        spread = {}
        for c in measures:
            v = pd.to_numeric(df[c.name], errors="coerce").dropna()
            if len(v) and (v >= 0).mean() > 0.95 and v.max() > 0:
                spread[c.name] = float(v.std() / (abs(v.mean()) + 1e-9))
        if spread:
            money = max(spread, key=spread.get)
    qty = next((c.name for c in measures if QTY_HINTS.search(c.name)), None)

    if not segs:
        warn.append("no grouping column found; segment analysis will be skipped.")
    if not texts:
        warn.append("no free-text column found; customer-language analysis will be skipped.")

    metrics = [{"key": "records", "label": "Record count", "kind": "count"}]
    if money:
        metrics += [
            {"key": "total_value", "label": f"Total {money}", "kind": "sum",
             "source": money},
            {"key": "avg_value", "label": f"Average {money}", "kind": "mean",
             "source": money},
        ]
    if qty and qty != money:
        metrics.append({"key": "total_quantity", "label": f"Total {qty}",
                        "kind": "sum", "source": qty})
    for c in measures:
        if c.name in (money, qty):
            continue
        metrics.append({"key": f"mean__{c.name}", "label": f"Average {c.name}",
                        "kind": "mean", "source": c.name})

    d = pd.to_datetime(df[date_col], errors="coerce")
    span = {"start": str(d.min())[:10], "end": str(d.max())[:10],
            "weeks": int(d.dt.to_period("W").nunique())}
    if span["weeks"] < 12:
        warn.append(f"only {span['weeks']} weeks of data; a baseline needs more "
                    f"history to be reliable.")

    return Profile(
        rows=len(df), columns=cols, date_column=date_col,
        segment_columns=[c.name for c in segs][:6],
        measure_columns=[c.name for c in measures],
        text_column=texts[0].name if texts else None,
        id_column=ids[0].name if ids else None,
        money_column=money, quantity_column=qty,
        metrics=metrics, span=span, warnings=warn,
    )


# --------------------------------------------------------------------------
# panel construction
# --------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def build_panel(df: pd.DataFrame, prof: Profile) -> pd.DataFrame:
    """Reshape an arbitrary table into the row-level panel the scopes expect."""
    p = df.copy()
    p["_date"] = pd.to_datetime(p[prof.date_column], errors="coerce")
    p = p[p["_date"].notna()].copy()
    if p.empty:
        raise ValueError("no rows had a readable date")
    p["week"] = p["_date"].dt.to_period("W").dt.start_time
    p["wk"] = p["week"].astype(str).str[:10]
    p["delivered"] = True                 # every row counts; no status concept here

    if prof.money_column:
        p["avg_value"] = pd.to_numeric(p[prof.money_column], errors="coerce")
    if prof.quantity_column:
        p["total_quantity"] = pd.to_numeric(p[prof.quantity_column], errors="coerce")
    for c in prof.measure_columns:
        if c not in (prof.money_column, prof.quantity_column):
            p[f"mean__{c}"] = pd.to_numeric(p[c], errors="coerce")

    if prof.text_column:
        p["_text"] = p[prof.text_column].fillna("").astype(str)
        p["has_text"] = p["_text"].str.len().gt(0)
        p["_norm"] = p["_text"].map(_strip_accents)
    else:
        p["has_text"] = False
    return p


def weekly_panel(p: pd.DataFrame, prof: Profile) -> pd.DataFrame:
    g = p.groupby("week")
    out = pd.DataFrame({"records": g.size()})
    out["orders"] = out["records"]        # scopes use `orders` for volume
    if prof.money_column:
        out["total_value"] = g["avg_value"].sum()
        out["avg_value"] = g["avg_value"].mean()
    if prof.quantity_column:
        out["total_quantity"] = g["total_quantity"].sum()
    for c in prof.measure_columns:
        if c not in (prof.money_column, prof.quantity_column):
            out[f"mean__{c}"] = g[f"mean__{c}"].mean()
    return out.reset_index()
