# mini_app.py — FOBOH Highlights UI (categories + frequency + score breakdown)
# Run:  python -m streamlit run mini_app.py

import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
HL_PATH = DATA_DIR / "highlights.json"

st.set_page_config(page_title="FOBOH – Highlights", layout="wide")
st.title("FOBOH – Daily Highlights")
st.caption("Ranked news highlights by category with frequency & score details.")

# ---------------- Utilities ----------------

def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        # allow various ISO-ish formats
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso

@st.cache_data(show_spinner=False)
def load_highlights(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Accept either {"cat":[...]} or flat list with "category"
    if isinstance(raw, dict):
        return {k: list(v) for k, v in raw.items()}
    if isinstance(raw, list):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in raw:
            cat = item.get("category") or "general"
            grouped.setdefault(cat, []).append(item)
        return grouped
    return {}

def run_cmd(args: list[str]) -> int:
    # Run a subprocess in the app root; return returncode
    try:
        res = subprocess.run(args, cwd=str(APP_ROOT), check=False, capture_output=True, text=True)
        if res.stdout:
            st.text(res.stdout)
        if res.stderr:
            st.text(res.stderr)
        return res.returncode
    except Exception as e:
        st.error(f"Failed to run: {' '.join(args)}\n{e}")
        return 1

# ---------------- Sidebar Controls ----------------

st.sidebar.header("Controls")

# Sorting and display
sort_by = st.sidebar.selectbox(
    "Sort highlights by",
    ["score_total (desc)", "sources_count (desc)", "cluster_size (desc)", "published_at (desc)"],
    index=0,
)
max_per_cat = st.sidebar.slider("Max items per category", 1, 20, 5, 1)
query_filter = st.sidebar.text_input("Filter in title/summary (optional)", "")

show_scores = st.sidebar.checkbox("Show score breakdown", value=True)
show_meta   = st.sidebar.checkbox("Show link & meta line", value=True)

# Admin: run steps 1–3 from UI
st.sidebar.header("Admin")
with st.sidebar.expander("Refresh highlights (Steps 1–3)", expanded=False):
    since_ingest = st.text_input("Ingest window (step1)", value="2d")
    limit_ingest = st.number_input("Ingest limit (step1)", min_value=50, max_value=2000, value=300, step=50)
    topk_export  = st.number_input("Top-K per category (step3)", min_value=1, max_value=20, value=5, step=1)
    if st.button("Run Steps 1–3"):
        with st.spinner("Step 1: Ingest feeds…"):
            run_cmd(["python", "step1_ingest.py", "--since", str(since_ingest), "--limit", str(limit_ingest)])
        with st.spinner("Step 2: Cluster / process…"):
            run_cmd(["python", "step2_process.py", "--since", "365d"])
        with st.spinner("Step 3: Rank highlights…"):
            run_cmd(["python", "step3_highlights.py", "--since", "365d", "--topk", str(topk_export)])
        st.success("Highlights updated. Click 'Reload highlights' below.")
        st.cache_data.clear()

# ---------------- Data Load ----------------

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Reload highlights"):
        st.cache_data.clear()

data = load_highlights(HL_PATH)
cats = sorted(list(data.keys()))
with col2:
    if HL_PATH.exists():
        ts = datetime.fromtimestamp(HL_PATH.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"Last generated: {ts}")
    else:
        st.warning("No highlights.json yet. Use the Admin panel to run Steps 1–3.")

if not data:
    st.stop()

# Category selection
sel_cats = st.multiselect("Categories to show", options=cats, default=cats)
if not sel_cats:
    st.info("Pick at least one category.")
    st.stop()

# ---------------- Rendering ----------------

def sort_key(item: Dict[str, Any]):
    if sort_by.startswith("score_total"):
        return float(item.get("score_total", item.get("score", 0.0)))
    if sort_by.startswith("sources_count"):
        return int(item.get("sources_count", item.get("freq_sources", 0)))
    if sort_by.startswith("cluster_size"):
        return int(item.get("cluster_size", 0))
    if sort_by.startswith("published_at"):
        # descending sort, so return a comparable numeric timestamp
        ts = item.get("published_at")
        try:
            dt = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0

for cat in sel_cats:
    st.markdown(f"## {cat.capitalize()}")
    items = list(data.get(cat, []))

    # sort (desc for all configured options)
    rev = True
    items.sort(key=sort_key, reverse=rev)

    # filter
    if query_filter.strip():
        q = query_filter.lower().strip()
        def _keep(it: Dict[str, Any]) -> bool:
            hay = f"{it.get('title','')} {it.get('summary','')}".lower()
            return q in hay
        items = [it for it in items if _keep(it)]

    # cap count
    items = items[:max_per_cat]

    if not items:
        st.info("No items for this category with current filters.")
        st.divider()
        continue

    # render each item
    for h in items:
        title = h.get("title", "(untitled)")
        url = h.get("url", "")
        src = h.get("source_domain", "")
        when = _fmt_date(h.get("published_at"))
        summary = h.get("summary", "")
        seen = h.get("sources_count", h.get("freq_sources", 0))
        clsz = h.get("cluster_size", 0)
        total = float(h.get("score_total", h.get("score", 0.0)))

        # header with link
        if url:
            st.markdown(f"### [{title}]({url})")
        else:
            st.markdown(f"### {title}")

        # meta line
        if show_meta:
            meta_line = " • ".join([p for p in [src or None, when or None] if p])
            if meta_line:
                st.caption(meta_line)

        # frequency line (explicit requirement)
        st.markdown(f"**Seen on {seen} sources • {clsz} related articles**")

        # summary
        if summary:
            st.write(summary)

        # score breakdown expander
        if show_scores:
            score_src  = h.get("score_src_domains", None)
            score_size = h.get("score_cluster_size", None)
            score_kw   = h.get("score_keywords", None)
            score_rec  = h.get("score_recency", None)
            with st.expander(f"Score details (total: {total:.2f})", expanded=False):
                st.markdown(
                    f"- Distinct sources: {score_src if score_src is not None else '—'}\n"
                    f"- Cluster size: {score_size if score_size is not None else '—'}\n"
                    f"- Keyword evidence: {score_kw if score_kw is not None else '—'}\n"
                    f"- Recency factor: {score_rec if score_rec is not None else '—'}"
                )

        st.divider()
