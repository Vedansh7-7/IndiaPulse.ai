"""Paths, decision thresholds, and the topic lexicon.

All tunable values live here so they can be reviewed in one place. Thresholds
set how strict a decision rule is; they never set the size of a result.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "out"

# ---- Decision thresholds (documented, auditable) -------------------------
ALPHA = 0.05              # significance level for individual tests
FDR_ALPHA = 0.05          # Benjamini-Hochberg false discovery rate
ROBUST_Z_TRIGGER = 3.0    # |robust z| beyond this = candidate anomaly
MIN_SEGMENT_N = 40        # below this a segment is too thin to conclude from
SEPARABILITY_MARGIN = 0.15  # top-2 evidence scores closer than this => Inconclusive
POWER = 0.80
BOOTSTRAP_N = 2000
SEED = 20260830

# ---- Analysis window -----------------------------------------------------
# Olist has negligible volume before 2017-01 and after 2018-08; restricting to
# the dense period avoids drawing conclusions from a handful of orders.
PANEL_START = "2017-01-02"
PANEL_END = "2018-08-20"

# ---- Topic lexicon (Portuguese, accent-stripped, lowercased) -------------
# Bare "prazo" is excluded from delivery_delay. It appears in
# both "chegou antes do prazo" (arrived early - positive) and "fora do prazo"
# (late - negative), so matching it alone inflates the delay topic with happy
# customers. Only the unambiguous negative forms are matched.
TOPIC_PATTERNS: dict[str, str] = {
    "delivery_delay": (
        r"atras|nao chegou|ainda nao receb|ainda nao chegou|nao receb|demor"
        r"|fora do prazo|nao foi entregue|ate agora nao|nada ate agora"
        r"|nao entregue|aguardando entrega"
    ),
    "product_quality": (
        r"quebrad|defeit|danific|estrag|pessima qualidade|de pessima"
        r"|nao funciona|veio ruim|maquita|material ruim|qualidade ruim"
    ),
    "wrong_item": (
        r"errad|diferente do|nao era o|outro produto|trocad|nao corresponde"
    ),
    "missing_item": (
        r"faltando|veio so|so veio|incompleto|faltou|apenas uma parte"
    ),
    "positive": (
        r"otim|excelent|adorei|recomend|perfeito|chegou antes|super rapid"
        r"|muito bom|amei|maravilhos|chegou rapido"
    ),
}

# Human-readable labels for the UI / narrative
TOPIC_LABELS = {
    "delivery_delay": "Delivery delay complaints",
    "product_quality": "Product quality complaints",
    "wrong_item": "Wrong item received",
    "missing_item": "Missing / incomplete order",
    "positive": "Positive sentiment",
}

STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapa",
    "BA": "Bahia", "CE": "Ceara", "DF": "Distrito Federal", "ES": "Espirito Santo",
    "GO": "Goias", "MA": "Maranhao", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Para", "PB": "Paraiba", "PE": "Pernambuco",
    "PI": "Piaui", "PR": "Parana", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondonia", "RR": "Roraima", "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina", "SE": "Sergipe", "SP": "Sao Paulo", "TO": "Tocantins",
}
