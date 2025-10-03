import json, math, re, sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HIGHLIGHTS_JSON = DATA_DIR / "highlights.json"
ARTICLES_DB = DATA_DIR / "articles.db"

st.set_page_config(page_title="FOBOH – ML Chatbot", layout="wide")
st.title("FOBOH – ML Chatbot")
st.caption("TF-IDF + BM25 + (optional) sentence embeddings with guardrails")

# ----------------- tokenization & hints -----------------

CATEGORY_HINTS = {
    "finance": {
        "inflation","cpi","price","cost","costofliving","interest","rate","rba","market","earnings",
        "economy","economic","gdp","recession","wage","unemployment","hike","cut","cash","bond"
    },
    "sports": {"afl","socceroo","football","soccer","cricket","match","win","qualifier","game","goal"},
    "entertainment": {"film","festival","movie","music","concert","tv","drama","celebrity","premiere"},
    "general": set(),
}

def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"): return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):  return token[:-2]
    if len(token) > 3 and token.endswith("s"):   return token[:-1]
    return token

def tokenize(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [singularize(t) for t in raw if t]

def guess_category(tokens: List[str]) -> Optional[str]:
    if not tokens: return None
    scores = {cat: len(set(tokens) & hints) for cat, hints in CATEGORY_HINTS.items()}
    best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_cat if best_score > 0 else None

# ----------------- data loading -----------------

def load_highlights(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists(): return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def corpus_from_highlights(hl: Dict[str, List[Dict[str, Any]]], cats: List[str]):
    docs, meta, tok_docs = [], [], []
    for cat in cats:
        for it in hl.get(cat, []):
            text = f"{it.get('title','')}. {it.get('summary','')} (Category: {cat})"
            docs.append(text)
            meta.append({"category": cat, **it})
            tok_docs.append(tokenize(text))
    return docs, meta, tok_docs

# ----------------- index builders -----------------

@st.cache_resource(show_spinner=False)
def build_tfidf(docs: List[str]):
    vec = TfidfVectorizer(analyzer=tokenize)
    X = vec.fit_transform(docs) if docs else None
    return vec, X

@st.cache_resource(show_spinner=False)
def build_bm25(tok_docs: List[List[str]]):
    return BM25Okapi(tok_docs) if tok_docs else None

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def build_embeddings(model_name: str, docs: List[str]):
    model = load_embedder(model_name)
    if model is None or not docs:
        return None
    embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)

# ----------------- retrieval -----------------

def tfidf_scores(vec: TfidfVectorizer, X, query: str) -> np.ndarray:
    if vec is None or X is None or not query: return np.zeros(X.shape[0] if X is not None else 0)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    return sims

def bm25_scores(bm25: BM25Okapi, query: str) -> np.ndarray:
    if bm25 is None or not query: return np.zeros(len(bm25.doc_len) if bm25 else 0)
    toks = tokenize(query)
    return np.array(bm25.get_scores(toks), dtype=float)

def embed_scores(embs_docs: Optional[np.ndarray], model_name: str, query: str) -> np.ndarray:
    if embs_docs is None or not query: return np.zeros(0)
    model = load_embedder(model_name)
    if model is None: return np.zeros(len(embs_docs))
    qvec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = np.dot(embs_docs, qvec)
    return sims

def minmax_scale(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    lo, hi = np.min(x), np.max(x)
    if hi <= lo: return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def fuse_scores(scores: List[np.ndarray], weights: List[float]) -> np.ndarray:
    # align lengths
    L = max((len(s) for s in scores if s is not None), default=0)
    if L == 0: return np.zeros(0)
    total = np.zeros(L, dtype=float)
    for s, w in zip(scores, weights):
        if s is None or len(s) == 0: continue
        s_norm = minmax_scale(s)
        total[:len(s_norm)] += w * s_norm
    return total

# ----------------- DB helpers -----------------

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

# ----------------- UI controls -----------------

highlights = load_highlights(HIGHLIGHTS_JSON)
if not highlights:
    st.warning("No highlights found. Run Steps 1–3 and ensure data/highlights.json exists.")
    st.stop()

categories = list(highlights.keys())
cat_filter = st.sidebar.multiselect("Restrict categories", categories, default=categories)

retriever = st.sidebar.selectbox("Retriever", ["Hybrid (Embeddings+TF-IDF+BM25)", "TF-IDF", "BM25", "Embeddings"])
embed_model = st.sidebar.text_input("Embeddings model", "sentence-transformers/all-MiniLM-L6-v2")
min_sim = st.sidebar.slider("Answer threshold", 0.0, 1.0, 0.12, 0.01)
topk = st.sidebar.slider("Top-K to consider", 1, 20, 5)

docs, meta, tok_docs = corpus_from_highlights(highlights, cat_filter)

vec, X = build_tfidf(docs)
bm25 = build_bm25(tok_docs)
embs = build_embeddings(embed_model, docs) if retriever in ("Embeddings", "Hybrid (Embeddings+TF-IDF+BM25)") else None

if retriever.startswith("Embedding") and embs is None:
    st.info("Embeddings not available (model missing or blocked). Falling back to TF-IDF + BM25.")
    retriever = "Hybrid (Embeddings+TF-IDF+BM25)"

# ----------------- Chat loop -----------------

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for role, msg in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(msg)

q = st.chat_input("Ask about today's highlights (e.g., 'finance inflation RBA')…")
if q:
    st.session_state.msgs.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        # category bias
        q_tokens = tokenize(q)
        cat_guess = guess_category(q_tokens)

        # compute scores
        s_tfidf = tfidf_scores(vec, X, q)
        s_bm25  = bm25_scores(bm25, q)
        s_emb   = embed_scores(embs, embed_model, q) if embs is not None else None

        # assemble according to retriever choice
        if retriever == "TF-IDF":
            combined = minmax_scale(s_tfidf)
        elif retriever == "BM25":
            combined = minmax_scale(s_bm25)
        elif retriever == "Embeddings":
            combined = minmax_scale(s_emb if s_emb is not None else np.zeros_like(s_tfidf))
        else:
            # Hybrid: prefer embeddings, then tfidf, then bm25
            combined = fuse_scores(
                [s_emb if s_emb is not None else np.zeros_like(s_tfidf), s_tfidf, s_bm25],
                [0.55, 0.3, 0.15],
            )

        # apply soft category bias if we guessed one and it's allowed by filter
        if cat_guess and cat_guess in cat_filter:
            bias = np.array([1.0 + 0.25 * (1 if m.get("category") == cat_guess else 0) for m in meta])
            combined = combined * bias

        # rank & guardrail
        order = np.argsort(-combined)[:max(topk, 3)]
        ranked = [(int(i), float(combined[i])) for i in order]
        # pick first with token overlap & above threshold
        def has_overlap(idx: int) -> bool:
            doc_text = f"{meta[idx].get('title','')} {meta[idx].get('summary','')}"
            return len(set(q_tokens) & set(tokenize(doc_text))) > 0

        chosen = None
        for i, sc in ranked:
            if sc >= min_sim and has_overlap(i):
                chosen = (i, sc); break

        if not ranked or chosen is None:
            st.markdown("I couldn’t find a good match. Try more detail (e.g., **'finance inflation RBA'**) or switch retriever in the sidebar.")
        else:
            i0, sc0 = chosen
            m0 = meta[i0]
            title = m0.get("title") or "(untitled)"
            summ  = m0.get("summary") or ""
            ans   = f"**{title}** — {summ}\n\n_Score: {sc0:.2f}_"

            dfc = fetch_cluster_articles(m0.get("cluster_id"))
            if not dfc.empty:
                srcs = dfc['source_domain'].dropna().value_counts().head(5)
                if not srcs.empty:
                    ans += "\n\nSources: " + ", ".join([f"{d} ({c})" for d, c in srcs.items()])
                link = dfc.loc[dfc['dedup_canonical'] == 1, 'url']
                if not link.empty:
                    ans += f"\n\nLink: {link.iloc[0]}"

            st.markdown(ans)

        # transparency table
        with st.expander("Top candidates"):
            rows = []
            for i, sc in ranked:
                rows.append({
                    "score": round(sc, 4),
                    "category": meta[i].get("category"),
                    "title": meta[i].get("title"),
                    "cluster_id": meta[i].get("cluster_id"),
                    "top_article_id": meta[i].get("top_article_id"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.session_state.msgs.append(("assistant", ""))
