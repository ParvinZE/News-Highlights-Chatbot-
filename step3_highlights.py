# step3_highlights.py — build ranked highlights per category and export for the UI
# - Uses per-keyword weighted bonus (e.g., breaking/exclusive/live/update)
# - Exposes frequency across sources and score breakdown
# - Writes top-K rows to the `highlights` table (backward compatible)
# - Exports rich JSON to data/highlights.json for the UI

import argparse, json, math, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from db import get_conn, init_db

# Try to import config; provide safe fallbacks if not present
try:
    from config import SUPPORTED_CATEGORIES as SUPPORTED_CATEGORIES_CFG
except Exception:
    SUPPORTED_CATEGORIES_CFG = ["finance", "sports", "music", "lifestyle"]

try:
    # May be list/tuple/set of keywords OR dict {keyword: weight}
    from config import KEYWORDS as CONFIG_KEYWORDS
except Exception:
    CONFIG_KEYWORDS = None

from dateutil import parser as dtp
from email.utils import parsedate_to_datetime

# ---------------- Scoring weights (tunable) ----------------
# Global weights for each observable signal; keyword bonus is computed per-keyword below.
WEIGHTS = {
    "src": 2.0,      # distinct source domains
    "size": 1.0,     # cluster size
    "kw": 1.0,       # multiplier for the per-keyword weighted bonus (keep 1.0 to avoid double-weighting)
    "recency": 2.0,  # freshness multiplier (half-life handled in recency_weight)
}

# Default per-keyword weights (used if CONFIG_KEYWORDS not provided)
DEFAULT_KEYWORD_WEIGHTS = {
    "breaking": 2.0,
    "exclusive": 1.5,
    "live": 1.2,
    "update": 1.0,
}

def _build_keyword_weights(cfg) -> Dict[str, float]:
    """Accept dict (keyword->weight) or iterable (uniform weight=1.0)."""
    if isinstance(cfg, dict):
        return {str(k).lower(): float(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple, set)):
        return {str(k).lower(): 1.0 for k in cfg}
    return dict(DEFAULT_KEYWORD_WEIGHTS)

KEYWORD_WEIGHTS = _build_keyword_weights(CONFIG_KEYWORDS)

# ---------------- Helpers ----------------

def _parse_dt_any(value: Optional[str]):
    """Parse many datetime formats safely; return timezone-aware datetime or None."""
    if not value:
        return None
    try:
        return dtp.parse(value)
    except Exception:
        try:
            return parsedate_to_datetime(value)
        except Exception:
            return None

def parse_since(value: Optional[str]) -> Optional[str]:
    """Accept '365d', '48h', '12m', '1y' or an ISO string; return ISO or None."""
    if not value:
        return None
    m = re.fullmatch(r"(\d+)([dhmy])", value.strip())
    if m:
        n, u = int(m.group(1)), m.group(2)
        if u == "h": delta = timedelta(hours=n)
        elif u == "d": delta = timedelta(days=n)
        elif u == "m": delta = timedelta(days=30 * n)
        else:          delta = timedelta(days=365 * n)
        return (datetime.now(timezone.utc) - delta).isoformat()
    # assume it's already ISO-like
    return value

def recency_weight(published_iso: Optional[str], half_life_hours: float = 48.0) -> float:
    """Exponential decay based on hours since publish; higher is fresher."""
    if not published_iso:
        return 0.8
    dt = _parse_dt_any(published_iso)
    if not dt:
        return 0.8
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_h = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    return 0.5 ** (age_h / half_life_hours)

def keyword_raw_hits(text: str) -> int:
    """Number of configured keywords present (for transparency in UI)."""
    t = (text or "").lower()
    return sum(1 for k in KEYWORD_WEIGHTS.keys() if k in t)

def keyword_bonus(text: str) -> float:
    """Weighted sum over configured keywords present in the text."""
    t = (text or "").lower()
    return sum(w for k, w in KEYWORD_WEIGHTS.items() if k in t)

# ---------------- Main logic ----------------

def build_highlights(categories: List[str], since: Optional[str], topk: int) -> Dict[str, List[Dict[str, Any]]]:
    """Compute ranked highlights per category and export combined JSON for the UI."""
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    combined: Dict[str, List[Dict[str, Any]]] = {}

    since_iso = parse_since(since)

    for cat in categories:
        # Pull canonical + non-canonical members so we can compute frequency & pick display representative
        q = """
SELECT
  cluster_id,
  id,
  title,
  summary_abs,
  source_domain,
  published_at,
  dedup_canonical,
  url
FROM articles
WHERE category_final = ? AND cluster_id IS NOT NULL
"""
        params = [cat]
        if since_iso:
            q += " AND (published_at IS NULL OR published_at >= ?)"
            params.append(since_iso)
        q += " ORDER BY cluster_id ASC, published_at DESC"

        rows = cur.execute(q, params).fetchall()

        # Group by cluster_id
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for (cid, aid, title, summary, src_dom, pub, is_canon, url) in rows:
            clusters.setdefault(int(cid), []).append({
                "article_id": int(aid),
                "title": title or "",
                "summary": summary or "",
                "src": (src_dom or ""),
                "pub": pub,
                "url": url or "",
                "canon": bool(is_canon),
            })

        # Score clusters and prepare ranked list
        ranked: List[Dict[str, Any]] = []
        for cid, items in clusters.items():
            if not items:
                continue

            # frequency across sources & cluster size
            sources = len({it["src"] for it in items if it.get("src")})
            size = len(items)

            # canonical representative (or most recent if none flagged)
            top = next((it for it in items if it["canon"]), items[0])

            # keyword & recency evidence
            text = f"{top['title']} {top['summary']}"
            kw_raw = keyword_raw_hits(text)      # raw count for UI
            kw_bns = keyword_bonus(text)         # weighted sum for scoring
            rw = recency_weight(top["pub"])

            # component scores (kept separate for UI breakdown)
            score_src  = WEIGHTS["src"]     * math.log1p(sources)
            score_size = WEIGHTS["size"]    * math.log1p(size)
            score_kw   = WEIGHTS["kw"]      * kw_bns
            score_rec  = WEIGHTS["recency"] * rw
            score_total = score_src + score_size + score_kw + score_rec

            ranked.append({
                "cluster_id": cid,
                "top_article_id": top["article_id"],
                "title": top["title"],
                "summary": top["summary"],

                # Link + metadata for the UI
                "url": top["url"],
                "source_domain": top["src"],
                "published_at": top["pub"],
                "category": cat,

                # Frequency fields (explicit per brief)
                "freq_sources": int(sources),     # back-compat name
                "sources_count": int(sources),    # clearer alias for UI
                "cluster_size": int(size),

                # Score components
                "score_src_domains": float(score_src),
                "score_cluster_size": float(score_size),
                "score_keywords": float(score_kw),
                "score_recency": float(score_rec),

                # Totals (keep original key, add alias)
                "score": float(score_total),
                "score_total": float(score_total),

                # raw evidence for transparency
                "keywords": int(kw_raw),
                "recency_weight": float(rw),
            })

        # Sort by total score (desc) and persist top-K to DB
        ranked.sort(key=lambda x: x["score_total"], reverse=True)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for it in ranked[:topk]:
            cur.execute(
                """
INSERT OR REPLACE INTO highlights
(category, cluster_id, top_article_id, score, freq_sources, cluster_size, keywords, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
                (
                    cat,
                    it["cluster_id"],
                    it["top_article_id"],
                    float(it["score_total"]),
                    int(it["freq_sources"]),
                    int(it["cluster_size"]),
                    int(it["keywords"]),
                    today_str,
                ),
            )
        conn.commit()

        # Keep top-K in the combined JSON export
        combined[cat] = ranked[:topk]

    # Export JSON for the UI
    outp = Path(__file__).resolve().parent / "data" / "highlights.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    conn.close()
    print(f"[step3] Exported combined highlights to {outp}")
    return combined

# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--categories",
        nargs="*",
        default=SUPPORTED_CATEGORIES_CFG,
        help="Categories to process (defaults to config.SUPPORTED_CATEGORIES)",
    )
    ap.add_argument(
        "--since",
        type=str,
        default=None,
        help="Time window like 365d/48h/12m/1y or ISO (e.g., 2024-01-01T00:00:00Z)",
    )
    ap.add_argument("--topk", type=int, default=5, help="Top K clusters per category")
    args = ap.parse_args()

    build_highlights(args.categories, args.since, args.topk)
