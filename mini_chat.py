import json, math, re, sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HIGHLIGHTS_JSON = DATA_DIR / "highlights.json"
ARTICLES_DB = DATA_DIR / "articles.db"

st.set_page_config(page_title="FOBOH – Mini Chatbot", layout="wide")
st.title("FOBOH – Mini Chatbot")
st.caption("Offline retrieval over highlights.json using a tiny built-in TF-IDF (no scikit-learn)")

def load_highlights(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if t]

def build_index(highlights: Dict[str, List[Dict[str, Any]]], cats: List[str]):
    docs, meta = [], []
    for cat in cats:
        for it in highlights.get(cat, []):
            text = f"{it.get('title','')}. {it.get('summary','')} (Category: {cat})"
            docs.append(text)
            meta.append({"category": cat, **it})
    if not docs:
        return None, None, []
    # document frequency (for IDF)
    df = Counter()
    tok_docs = []
    for d in docs:
        toks = tokenize(d)
        tok_docs.append(toks)
        df.update(set(toks))
    N = len(docs)
    # build TF-IDF vectors as dicts
    vectors = []
    for toks in tok_docs:
        tf = Counter(toks)
        vec = {w: (tf[w] / len(toks)) * math.log((N + 1) / (df[w] + 1)) for w in tf}
        vectors.append(vec)
    return {"vectors": vectors, "df": df, "N": N}, meta, docs

def cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
    n1 = math.sqrt(sum(x * x for x in v1.values()))
    n2 = math.sqrt(sum(x * x for x in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def vec_query(q: str, df: Counter, N: int):
    toks = tokenize(q)
    if not toks:
        return {}
    tf = Counter(toks)
    return {w: (tf[w] / len(toks)) * math.log((N + 1) / (df.get(w, 0) + 1)) for w in tf}

def fetch_cluster_articles(cluster_id: int) -> pd.DataFrame:
    if not ARTICLES_DB.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(ARTICLES_DB)
        q = """
SELECT id as article_id, title, source, source_domain, url, published_at, dedup_canonical
FROM articles
WHERE cluster_id = ?
ORDER BY published_at DESC
"""
        df = pd.read_sql_query(q, conn, params=(int(cluster_id),))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# ---- App ----
highlights = load_highlights(HIGHLIGHTS_JSON)
if not highlights:
    st.warning("No highlights found. Ensure data/highlights.json exists (export from Step 3).")
    st.stop()

categories = list(highlights.keys())
cat_filter = st.sidebar.multiselect("Restrict to categories", categories, default=categories)

index, meta, _ = build_index(highlights, cats=cat_filter)
if index is None:
    st.warning("No documents to search.")
    st.stop()

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for role, msg in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(msg)

q = st.chat_input("Ask about today's highlights…")
if q:
    st.session_state.msgs.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        qv = vec_query(q, index["df"], index["N"])
        sims = [(i, cosine(qv, index["vectors"][i])) for i in range(len(meta))]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:3]
        if not top or top[0][1] <= 0.0:
            ans = "I couldn't find a relevant highlight. Try rephrasing or narrow the categories."
        else:
            i0, _ = top[0]
            m0 = meta[i0]
            title = m0.get("title") or "(untitled)"
            summ = m0.get("summary") or ""
            ans = f"**{title}** — {summ}"
            dfc = fetch_cluster_articles(m0.get("cluster_id"))
            if not dfc.empty:
                srcs = dfc['source_domain'].dropna().value_counts().head(5)
                if not srcs.empty:
                    ans += "\n\nSources: " + ", ".join([f"{d} ({c})" for d, c in srcs.items()])
                link = dfc.loc[dfc['dedup_canonical'] == 1, 'url']
                if not link.empty:
                    ans += f"\n\nLink: {link.iloc[0]}"
        st.markdown(ans)
        with st.expander("Top matches"):
            rows = []
            for i, sc in top:
                rows.append({
                    "score": round(sc, 4),
                    "category": meta[i].get("category"),
                    "title": meta[i].get("title"),
                    "cluster_id": meta[i].get("cluster_id"),
                    "top_article_id": meta[i].get("top_article_id"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.session_state.msgs.append(("assistant", ans))

st.caption("© FOBOH – mini TF-IDF chatbot (no external ML libs)")
