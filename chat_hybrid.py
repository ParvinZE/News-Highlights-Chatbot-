# chat_hybrid.py — Hybrid keyword + semantic chatbot with:
#   • BM25 shortlist + sentence-embedding fusion
#   • Cross-Encoder re-rank
#   • Temporal bias (today / this week)
#   • Sports phrase boosts + improved category detection
#   • "List mode" for exploratory queries (any/what's on/this week?) with feedback UI
#   • In-app feedback + evaluation (no CSV)
#   • Optional Learn-to-Rank (LightGBM or Logistic Regression)
#
# Run (after Steps 1–3 have produced data/highlights.json):
#   source .venv/bin/activate
#   python -m pip install --upgrade pip wheel
#   python -m pip install streamlit pandas numpy rank-bm25
#   python -m pip install sentence-transformers torch              # optional but recommended
#   python -m pip install lightgbm scikit-learn joblib             # optional for LTR
#   python -m streamlit run chat_hybrid.py

import json, re, sqlite3, unicodedata, os, time, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HIGHLIGHTS_JSON = DATA_DIR / "highlights.json"
ARTICLES_DB = DATA_DIR / "articles.db"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="FOBOH – Hybrid Chatbot (CE + Feedback + LTR)", layout="wide")
st.title("FOBOH – Hybrid (Keyword + Semantic) with Cross-Encoder, Feedback & LTR")
st.caption("BM25 shortlist → Embeddings fusion → Cross-Encoder re-rank; temporal & sports boosts; feedback-driven eval; optional LTR.")

# ---------- Tokenization, hints, expansion ----------
STOP = set("""
the of and a an to in on for with at from by as are was were be been being it its this that those these there here
how what when where who why which today todays tonight
""".split())

def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"): return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):  return token[:-2]
    if len(token) > 3 and token.endswith("s"):   return token[:-1]
    return token

def tokenize(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text or "").lower()
    raw = re.findall(r"[a-z0-9]+", text)
    return [singularize(t) for t in raw if t and t not in STOP]

CATEGORY_HINTS = {
    "finance": {
        "finance","financial","economy","economic","inflation","cpi","price","cost","costofliving","interest","rate","rates",
        "rba","reserve","bank","cash","market","markets","asx","shares","stocks","gdp","recession","employment","unemployment",
        "wage","bond","currency","aud","dollar","budget"
    },
    "sports": {
        "sport","sports","afl","nrl","a-league","w-league","aflw","soccer","football","cricket","rugby",
        "tennis","golf","basketball","nba","nfl","ufc","boxing","f1","formula","motogp","marathon",
        "match","game","fixture","round","qualifier","semi","semifinal","quarterfinal","final","finals",
        "grandfinal","derby","series","test","cup","worldcup","olympics","paralympics","commonwealth","open","masters"
    },
    "entertainment": {"film","festival","movie","music","concert","tv","drama","celebrity","premiere"},
    "general": set(),
}
AU_HINTS = {"australia","australian","au","rba","asx","sydney","melbourne","canberra","brisbane","perth","adelaide"}

def guess_category(tokens: List[str]) -> Optional[str]:
    if not tokens: return None
    scores = {cat: len(set(tokens) & hints) for cat, hints in CATEGORY_HINTS.items()}
    cat, sc = max(scores.items(), key=lambda kv: kv[1])
    if sc > 0:
        return cat
    if "sport" in tokens or "sports" in tokens: return "sports"
    if "finance" in tokens or "economic" in tokens or "economy" in tokens: return "finance"
    return None

def expand_query_tokens(tokens: List[str]) -> List[str]:
    tset = set(tokens)
    if "finance" in tokens or (set(tokens) & CATEGORY_HINTS["finance"]):
        tset |= CATEGORY_HINTS["finance"]
    if "australia" in tokens or "australian" in tokens or (set(tokens) & AU_HINTS):
        tset |= AU_HINTS
    if "sport" in tokens or "sports" in tokens or (set(tokens) & CATEGORY_HINTS["sports"]):
        tset |= CATEGORY_HINTS["sports"]
    return list(tset)

def detect_temporal_intent(q: str) -> Optional[str]:
    s = q.lower()
    if "today" in s or "tonight" in s or "this evening" in s: return "today"
    if "tomorrow" in s: return "tomorrow"
    if "this week" in s or "weekend" in s or "this weekend" in s: return "week"
    return None

def wants_list(q: str) -> bool:
    ql = q.lower()
    triggers = ["any", "what's on", "whats on", "what is on", "this week", "upcoming", "schedule", "fixtures"]
    return any(t in ql for t in triggers)

SPORT_PHRASES = {
    "grand final": 0.18, "cup final": 0.12, "finals": 0.10, "semi final": 0.12, "semifinal": 0.12,
    "qualifier": 0.08, "world cup": 0.15, "olympics": 0.14, "derby": 0.07, "match": 0.05,
    "round": 0.05, "test": 0.06, "series": 0.06, "open": 0.07, "tournament": 0.08
}
def phrase_boost(text: str) -> float:
    t = re.sub(r"\s+", " ", (text or "").lower())
    return sum(w for p, w in SPORT_PHRASES.items() if p in t)

def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo: return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_highlights() -> Dict[str, List[Dict[str, Any]]]:
    if not HIGHLIGHTS_JSON.exists(): return {}
    return json.loads(HIGHLIGHTS_JSON.read_text())

def corpus_from_highlights(hl: Dict[str, List[Dict[str, Any]]], cats: List[str]):
    docs, meta, tok_docs, titles, summaries = [], [], [], [], []
    for cat in cats:
        for it in hl.get(cat, []):
            t = it.get('title',''); s = it.get('summary','')
            text = f"{t}. {s} (Category: {cat})"
            docs.append(text)
            meta.append({"category": cat, **it})
            tok_docs.append(tokenize(text))
            titles.append(t); summaries.append(s)
    return docs, meta, tok_docs, titles, summaries

# ---------- Keyword: BM25 ----------
BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def build_bm25(tok_docs: List[List[str]]):
    if not BM25_AVAILABLE or not tok_docs: return None
    return BM25Okapi(tok_docs)

def bm25_scores_tokens(bm25, q_tokens: List[str], n_docs: int) -> np.ndarray:
    if bm25 is None or not q_tokens:
        return np.zeros(n_docs, dtype=float)
    try:
        return np.array(bm25.get_scores(q_tokens), dtype=float)
    except Exception:
        return np.zeros(n_docs, dtype=float)

# ---------- Semantic: Sentence Embeddings ----------
def try_load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def build_embeddings(model_name: str, docs: List[str]) -> Tuple[Optional[np.ndarray], Optional[object]]:
    model = try_load_embedder(model_name)
    if model is None or not docs: return None, None
    embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32), model

def embed_query(model, text: str) -> Optional[np.ndarray]:
    if model is None or not text: return None
    q = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(q, dtype=np.float32)

# ---------- Cross-Encoder re-ranker ----------
def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception:
        return None

def rerank_cross_encoder(cross_encoder, query: str, docs_texts: List[str]) -> List[float]:
    if cross_encoder is None or not docs_texts: return [0.0]*len(docs_texts)
    pairs = [(query, t) for t in docs_texts]
    import numpy as np
    return list(np.asarray(cross_encoder.predict(pairs), dtype=float))

# ---------- LTR: load/save, feature construction ----------
def load_ltr_model():
    model_path = MODELS_DIR / "ltr_model.pkl"
    if not model_path.exists(): return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        return None

def save_ltr_model(model):
    try:
        import joblib
        joblib.dump(model, MODELS_DIR / "ltr_model.pkl")
    except Exception:
        pass

def get_doc_features(meta_item: Dict[str, Any]) -> Dict[str, float]:
    return {
        "freq_sources": float(meta_item.get("freq_sources") or 0),
        "cluster_size": float(meta_item.get("cluster_size") or 0),
        "keywords": float(meta_item.get("keywords") or 0),
        "recency": float(meta_item.get("recency_weight") or 0),
    }

def build_features(query: str, q_tokens_raw: List[str], meta: List[Dict[str,Any]],
                   bm25, emb_model, embs_docs, base_scores: np.ndarray) -> np.ndarray:
    s_kw = bm25_scores_tokens(bm25, expand_query_tokens(q_tokens_raw), len(meta))
    if emb_model is not None and embs_docs is not None:
        qv = embed_query(emb_model, query)
        s_sem = np.dot(embs_docs, qv) if qv is not None else np.zeros_like(s_kw)
    else:
        s_sem = np.zeros_like(s_kw)

    qset = set(q_tokens_raw)
    rows = []
    cat_guess = guess_category(q_tokens_raw)
    for i, m in enumerate(meta):
        feats = get_doc_features(m)
        text = f"{m.get('title','')} {m.get('summary','')}"
        dset = set(tokenize(text))
        rows.append([
            s_kw[i],                                  # 0 BM25
            s_sem[i],                                 # 1 Emb cosine
            jaccard(qset, dset),                      # 2 token overlap
            1.0 if m.get("category") == cat_guess else 0.0,  # 3 cat match
            feats["freq_sources"],                    # 4
            feats["cluster_size"],                    # 5
            feats["keywords"],                        # 6
            feats["recency"],                         # 7
            base_scores[i],                           # 8 current fused (or CE) score
        ])
    return np.array(rows, dtype=float)

# ---------- Feedback table & gold building ----------
def ensure_feedback_table():
    if not ARTICLES_DB.exists():
        conn = sqlite3.connect(ARTICLES_DB); conn.execute("VACUUM"); conn.close()
    conn = sqlite3.connect(ARTICLES_DB)
    conn.execute("""
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  query TEXT NOT NULL,
  chosen_cluster INTEGER,
  correct_cluster INTEGER,
  category_guess TEXT,
  retriever TEXT,
  satisfied INTEGER
)""")
    conn.commit(); conn.close()

def log_feedback(query: str, chosen_cluster: Optional[int], category_guess: Optional[str],
                 retriever: str, satisfied: Optional[int], correct_cluster: Optional[int] = None):
    ensure_feedback_table()
    conn = sqlite3.connect(ARTICLES_DB)
    conn.execute(
        "INSERT INTO feedback(ts, query, chosen_cluster, correct_cluster, category_guess, retriever, satisfied) VALUES(datetime('now'),?,?,?,?,?,?)",
        (query, chosen_cluster, correct_cluster, category_guess, retriever, satisfied)
    )
    conn.commit(); conn.close()

def build_gold_from_feedback() -> pd.DataFrame:
    if not ARTICLES_DB.exists():
        return pd.DataFrame(columns=["query", "correct_cluster"])
    conn = sqlite3.connect(ARTICLES_DB)
    fb = pd.read_sql_query("SELECT * FROM feedback ORDER BY ts DESC", conn)
    conn.close()
    if fb.empty:
        return pd.DataFrame(columns=["query", "correct_cluster"])

    def pick_label(r):
        if pd.notna(r.get("correct_cluster")):
            try: return int(r["correct_cluster"])
            except Exception: return None
        if pd.notna(r.get("chosen_cluster")) and (r.get("satisfied", None) == 1):
            try: return int(r["chosen_cluster"])
            except Exception: return None
        return None

    fb["gold_cluster"] = fb.apply(pick_label, axis=1)
    gold = fb.dropna(subset=["gold_cluster"])[["query", "gold_cluster"]].copy()
    gold = gold.drop_duplicates(subset=["query"], keep="first")
    gold = gold.rename(columns={"gold_cluster": "correct_cluster"}).astype({"correct_cluster": int})
    return gold

# ---------- Metrics ----------
def ndcg_at_k(rels: List[float], k: int=5) -> float:
    rels = rels[:k]
    dcg = sum((rels[i] / math.log2(i+2)) for i in range(len(rels)))
    ideal = sorted(rels, reverse=True)
    idcg = sum((ideal[i] / math.log2(i+2)) for i in range(len(ideal)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(hits: List[int], k: int=5) -> float:
    for i, h in enumerate(hits[:k]):
        if h: return 1.0 / (i+1)
    return 0.0

# ---------- Load data & indices ----------
hl = load_highlights()
if not hl:
    st.warning("No highlights found. Run Steps 1–3 so data/highlights.json exists.")
    st.stop()

cats_all = list(hl.keys())
cats_sel = st.sidebar.multiselect("Categories", options=cats_all, default=cats_all)
docs, meta, tok_docs, titles, summaries = corpus_from_highlights(hl, cats_sel)
if not docs:
    st.warning("No documents in selected categories."); st.stop()

bm25 = build_bm25(tok_docs)

# ---------- Sidebar controls ----------
st.sidebar.subheader("Retrieval")
model_name = st.sidebar.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
use_semantic = st.sidebar.checkbox("Use semantic embeddings", value=True)
w_kw = st.sidebar.slider("Weight: keyword (BM25)", 0.0, 1.0, 0.6, 0.05)
w_sem = st.sidebar.slider("Weight: semantic", 0.0, 1.0, 0.4, 0.05)
shortlist_k = st.sidebar.slider("Keyword shortlist K", 5, 100, 30, 5)
min_sim = st.sidebar.slider("Answer threshold", 0.00, 1.00, 0.15, 0.01)
hard_filter_category = st.sidebar.checkbox("Hard-filter by guessed category", value=True)
location_bias = st.sidebar.checkbox("Boost Australia context when asked", value=True)

st.sidebar.subheader("Cross-Encoder")
use_ce = st.sidebar.checkbox("Enable cross-encoder re-rank", value=True)
ce_weight = st.sidebar.slider("Cross-encoder weight (vs fused)", 0.0, 1.0, 0.80, 0.05)

st.sidebar.subheader("Learn-to-Rank")
use_ltr = st.sidebar.checkbox("Use LTR model if available", value=True)
ltr_weight = st.sidebar.slider("LTR weight (vs previous)", 0.0, 1.0, 0.80, 0.05)

# ---------- Build embeddings & cross-encoder ----------
embs_docs, emb_model = (None, None)
if use_semantic:
    with st.spinner("Preparing embeddings…"):
        embs_docs, emb_model = build_embeddings(model_name, docs)

if "cross_encoder" not in st.session_state:
    st.session_state.cross_encoder = load_cross_encoder()
ce_model = st.session_state.cross_encoder

ltr_model = load_ltr_model() if use_ltr else None

# ---------- Chat history ----------
if "msgs" not in st.session_state:
    st.session_state.msgs = []

for role, msg in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(msg)

# ---------- Retrieval pipeline ----------
def compute_scores(query: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    q_tokens_raw = tokenize(query)
    q_tokens = expand_query_tokens(q_tokens_raw)
    cat_guess = guess_category(q_tokens_raw)

    # 1) Keyword shortlist
    s_kw = bm25_scores_tokens(bm25, q_tokens, len(meta))
    order_kw = np.argsort(-s_kw)[:max(shortlist_k, 5)]
    shortlisted = set(order_kw.tolist())

    # 2) Semantic (only within shortlist)
    s_sem = np.zeros_like(s_kw)
    if use_semantic and emb_model is not None and embs_docs is not None:
        qv = embed_query(emb_model, " ".join(q_tokens))
        if qv is not None:
            sims_all = np.dot(embs_docs, qv)
            s_sem = np.where(np.isin(np.arange(len(s_kw)), list(shortlisted)), sims_all, 0.0)

    # 3) Fuse keyword + semantic
    fused = w_kw * minmax(s_kw) + (w_sem * minmax(s_sem) if use_semantic and embs_docs is not None else 0.0)

    # 4) Guards: category + AU context
    combined = fused.copy()
    if cat_guess and hard_filter_category and cat_guess in cats_sel:
        for i, m in enumerate(meta):
            if m.get("category") != cat_guess:
                combined[i] = 0.0
    else:
        for i, m in enumerate(meta):
            if cat_guess and m.get("category") == cat_guess:
                combined[i] *= 1.15

    if location_bias and (set(q_tokens) & AU_HINTS):
        for i, m in enumerate(meta):
            doc_text = f"{m.get('title','')} {m.get('summary','')}"
            has_au = len(set(tokenize(doc_text)) & AU_HINTS) > 0
            if not has_au:
                combined[i] *= 0.7

    # 4.5) Temporal bias
    t_intent = detect_temporal_intent(query)
    if t_intent:
        rec = np.array([float(meta[i].get("recency_weight") or 0.0) for i in range(len(meta))])
        if t_intent == "today":
            combined *= (0.15 + 0.85 * rec)
        elif t_intent in ("tomorrow", "week"):
            combined *= (0.50 + 0.50 * rec)

    # 4.6) Sports phrase boosts
    if cat_guess == "sports":
        for i, m in enumerate(meta):
            cand = f"{m.get('title','')} {m.get('summary','')}"
            combined[i] += phrase_boost(cand)

    # 5) Cross-Encoder re-rank on top-N
    if use_ce and ce_model is not None:
        topN = [int(i) for i in np.argsort(-combined)[:20]]
        cand_texts = [f"{meta[i].get('title','')}. {meta[i].get('summary','')}" for i in topN]
        ce_scores = rerank_cross_encoder(ce_model, query, cand_texts)
        ce_norm = minmax(np.array(ce_scores, dtype=float))
        combined_ce = combined.copy()
        for rank, idx in enumerate(topN):
            combined_ce[idx] = ce_weight * ce_norm[rank] + (1.0 - ce_weight) * combined[idx]
        combined = combined_ce

    # 6) Learn-to-Rank (if available)
    if ltr_model is not None:
        X = build_features(query, q_tokens_raw, meta, bm25, emb_model, embs_docs, combined)
        try:
            ltr_pred = np.array(ltr_model.predict(X), dtype=float).reshape(-1)
            ltr_norm = minmax(ltr_pred)
            combined = ltr_weight * ltr_norm + (1.0 - ltr_weight) * combined
        except Exception:
            pass

    return combined, {"q_tokens_raw": q_tokens_raw, "cat_guess": cat_guess}

# ---------- DB: cluster articles ----------
def fetch_cluster_articles(cluster_id: int) -> pd.DataFrame:
    if not ARTICLES_DB.exists(): return pd.DataFrame()
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

# ---------- Chat loop ----------
q = st.chat_input("Ask about today's highlights (e.g., 'finance inflation RBA', 'AFL grand final this week')…")
if q:
    st.session_state.msgs.append(("user", q))
    with st.chat_message("user"): st.markdown(q)

    with st.chat_message("assistant"):
        combined, info = compute_scores(q)
        order = np.argsort(-combined)

        # LIST MODE: show several sports items for exploratory queries + feedback UI
        if wants_list(q) and (info["cat_guess"] == "sports" or "sport" in tokenize(q) or "sports" in tokenize(q)):
            picks = []
            qtok = set(tokenize(q))
            for i in order[:30]:
                doc_text = f"{meta[i].get('title','')} {meta[i].get('summary','')}"
                if len(qtok & set(tokenize(doc_text))) == 0:
                    continue
                if combined[i] < min_sim:
                    continue
                if meta[i].get("category") != "sports":
                    continue
                picks.append(i)
                if len(picks) >= 5:
                    break

            if not picks:
                st.markdown("I couldn’t find good sports matches for this week. Try adding a league or event name (e.g., **AFL Grand Final, World Cup qualifier, cricket series**).")
                log_feedback(query=q, chosen_cluster=None, category_guess=info["cat_guess"], retriever="hybrid_ce_ltr", satisfied=None)
            else:
                st.markdown("**Top sports highlights**")
                label_options = []
                for i in picks:
                    m0 = meta[i]
                    t0 = m0.get("title") or "(untitled)"
                    s0 = m0.get("summary") or ""
                    out = f"- **{t0}** — {s0}  \n  _Score: {combined[i]:.2f}_"
                    dfc = fetch_cluster_articles(m0.get("cluster_id"))
                    if not dfc.empty:
                        link = dfc.loc[dfc['dedup_canonical'] == 1, 'url']
                        if not link.empty:
                            out += f"\n  [Link]({link.iloc[0]})"
                    st.markdown(out)
                    label_options.append((f"{t0[:80]} … (cluster {m0.get('cluster_id')})", int(m0.get("cluster_id"))))

                ensure_feedback_table()
                st.divider()
                st.markdown("**Was one of these the correct event?**")
                if label_options:
                    labels_text = [lt for lt, _ in label_options]
                    labels_ids  = {lt: cid for lt, cid in label_options}
                    sel = st.radio("Select the correct event", labels_text, index=0, key=f"list_sel_{time.time()}")
                    c1, c2 = st.columns([1,1])
                    if c1.button("💾 Save correct event", key=f"save_list_{time.time()}"):
                        cid = labels_ids.get(sel)
                        log_feedback(query=q, chosen_cluster=cid, category_guess=info["cat_guess"],
                                     retriever="hybrid_ce_ltr", satisfied=1, correct_cluster=cid)
                        try: st.toast("Saved your label. Thanks!", icon="✅")
                        except Exception: st.success("Saved your label. Thanks!")
                    if c2.button("👎 Not helpful", key=f"down_list_{time.time()}"):
                        log_feedback(query=q, chosen_cluster=None, category_guess=info["cat_guess"],
                                     retriever="hybrid_ce_ltr", satisfied=0)
                        try: st.toast("Logged not-helpful.", icon="⚠️")
                        except Exception: st.warning("Logged not-helpful.")

        else:
            # SINGLE-ANSWER MODE
            chosen = None
            qset = set(info["q_tokens_raw"])
            def overlap_ok(idx: int) -> bool:
                t = f"{meta[idx].get('title','')} {meta[idx].get('summary','')}"
                return len(qset & set(tokenize(t))) > 0

            for i in order[:20]:
                if combined[i] >= min_sim and overlap_ok(i):
                    chosen = i; break

            if chosen is None:
                st.markdown("I couldn’t find a good match. Try adding a concrete term (e.g., **AFL, qualifier, grand final, World Cup**).")
                log_feedback(query=q, chosen_cluster=None, category_guess=info["cat_guess"], retriever="hybrid_ce_ltr", satisfied=None)
            else:
                m0 = meta[chosen]
                title = m0.get("title") or "(untitled)"
                summ  = m0.get("summary") or ""
                ans   = f"**{title}** — {summ}\n\n_Score: {combined[chosen]:.2f}_"
                dfc = fetch_cluster_articles(m0.get("cluster_id"))
                if not dfc.empty:
                    srcs = dfc['source_domain'].dropna().value_counts().head(5)
                    if not srcs.empty:
                        ans += "\n\nSources: " + ", ".join([f"{d} ({c})" for d, c in srcs.items()])
                    link = dfc.loc[dfc['dedup_canonical'] == 1, 'url']
                    if not link.empty:
                        ans += f"\n\nLink: {link.iloc[0]}"
                st.markdown(ans)

                ensure_feedback_table()
                c1, c2, _ = st.columns([1,1,2])
                if c1.button("👍 Helpful", key=f"up_{time.time()}"):
                    log_feedback(query=q, chosen_cluster=m0.get("cluster_id"), category_guess=info["cat_guess"],
                                 retriever="hybrid_ce_ltr", satisfied=1)
                    try: st.toast("Thanks! Logged your feedback.", icon="✅")
                    except Exception: st.success("Thanks! Logged your feedback.")
                if c2.button("👎 Not helpful", key=f"down_{time.time()}"):
                    log_feedback(query=q, chosen_cluster=m0.get("cluster_id"), category_guess=info["cat_guess"],
                                 retriever="hybrid_ce_ltr", satisfied=0)
                    try: st.toast("Logged not-helpful. You can label the correct item below.", icon="⚠️")
                    except Exception: st.warning("Logged not-helpful. You can label the correct item below.")

                # Let user mark the correct cluster from top results
                top_rows = []
                for rank_i, i in enumerate(order[:20], start=1):
                    top_rows.append({
                        "rank": rank_i,
                        "cluster_id": meta[i].get("cluster_id"),
                        "category": meta[i].get("category"),
                        "title": meta[i].get("title"),
                    })
                df_top = pd.DataFrame(top_rows)
                with st.expander("Top candidates"):
                    st.dataframe(df_top, use_container_width=True, hide_index=True)
                    idx = st.number_input("If the correct item is in the table above, enter its cluster_id and click Save:",
                                          value=0, step=1)
                    if st.button("Save correct cluster"):
                        if int(idx) != 0:
                            log_feedback(query=q, chosen_cluster=m0.get("cluster_id"), category_guess=info["cat_guess"],
                                         retriever="hybrid_ce_ltr", satisfied=1, correct_cluster=int(idx))
                            try: st.toast("Saved correct cluster label.", icon="💾")
                            except Exception: st.success("Saved correct cluster label.")

        st.session_state.msgs.append(("assistant", ""))

# ---------- Evaluation + LTR + Feedback tabs ----------
st.divider()
tabs = st.tabs(["📊 Evaluation (from feedback)", "🧠 Train LTR", "🗂 Feedback Log"])

with tabs[0]:
    st.subheader("Evaluation from in-app feedback")
    ensure_feedback_table()
    gold = build_gold_from_feedback()

    # Warn if current category filter hides labeled clusters
    present_clusters = {int(m["cluster_id"]) for m in meta if m.get("cluster_id") is not None}
    gold_visible = gold[gold["correct_cluster"].isin(present_clusters)].copy()

    if gold.empty:
        st.info("In chat, click 👍 Helpful or save a correct cluster; those become evaluation labels.")
    elif gold_visible.empty and not gold.empty:
        st.warning("You have labels, but none match the clusters in the currently selected categories. Select all categories in the sidebar and try again.")
    else:
        p_at1, mrrs, ndcgs = [], [], []
        rows = []
        for _, row in gold_visible.iterrows():
            qg   = str(row["query"])
            tgt  = int(row["correct_cluster"])
            scores, _ = compute_scores(qg)
            order = np.argsort(-scores)[:20]
            ranked_cids = [int(meta[i].get("cluster_id")) for i in order]
            rels = [1.0 if cid == tgt else 0.0 for cid in ranked_cids]
            p_at1.append(1.0 if rels[:1] and rels[0] == 1.0 else 0.0)
            mrrs.append(mrr_at_k(rels, 5))
            ndcgs.append(ndcg_at_k(rels, 5))
            rows.append({
                "query": qg,
                "correct_cluster": tgt,
                "top1_cluster": ranked_cids[0] if ranked_cids else None,
                "hit@1": bool(rels[:1] and rels[0] == 1.0),
                "MRR@5": round(mrr_at_k(rels, 5), 3),
                "nDCG@5": round(ndcg_at_k(rels, 5), 3),
            })
        st.write(f"**Queries evaluated:** {len(gold_visible)}")
        st.write(f"**P@1:** {np.mean(p_at1):.3f}  |  **MRR@5:** {np.mean(mrrs):.3f}  |  **nDCG@5:** {np.mean(ndcgs):.3f}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Labels come from your in-app feedback: explicit correct_cluster, or 👍 Helpful on the chosen result.")

with tabs[1]:
    st.subheader("Train Learn-to-Rank model")
    ensure_feedback_table()
    conn = sqlite3.connect(ARTICLES_DB)
    fb = pd.read_sql_query("SELECT * FROM feedback ORDER BY ts DESC", conn)
    conn.close()
    if fb.empty:
        st.info("No feedback yet. Ask a few questions and click 👍/👎 to collect data.")
    else:
        st.write(f"Feedback rows: {len(fb)}")
        # Build training set from feedback-derived gold
        gold = build_gold_from_feedback()
        if gold.empty:
            st.info("You need at least some positive labels (correct clusters) from feedback to train.")
        else:
            X_list, y_list, qid_list = [], [], []
            qid = 0
            for _, r in gold.iterrows():
                q_text = r["query"]; tgt = int(r["correct_cluster"])
                scores, _ = compute_scores(q_text)
                order = np.argsort(-scores)[:30]  # candidate set
                X = build_features(q_text, tokenize(q_text), meta, bm25, emb_model, embs_docs, scores)
                for i in order:
                    cid = int(meta[i].get("cluster_id"))
                    lbl = 1 if cid == tgt else 0
                    X_list.append(list(X[i]))
                    y_list.append(lbl)
                    qid_list.append(qid)
                qid += 1

            if not X_list:
                st.info("Not enough training pairs yet. Provide some positive labels first.")
            else:
                X = np.array(X_list, dtype=float); y = np.array(y_list, dtype=int); qids = np.array(qid_list, dtype=int)
                trainer = st.radio("Trainer", ["LightGBM LambdaRank", "Logistic Regression"], index=0)
                model = None
                if trainer.startswith("LightGBM"):
                    try:
                        import lightgbm as lgb
                        _, counts = np.unique(qids, return_counts=True)
                        train = lgb.Dataset(X, label=y, group=counts)
                        params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1,3,5],
                                      learning_rate=0.05, num_leaves=31, min_data_in_leaf=10)
                        booster = lgb.train(params, train, num_boost_round=300, valid_sets=[], verbose_eval=False)
                        class LGBWrapper:
                            def __init__(self, booster): self.booster = booster
                            def predict(self, X_): return self.booster.predict(X_)
                        model = LGBWrapper(booster)
                        save_ltr_model(model)
                        st.success("Trained LightGBM LambdaRank and saved to data/models/ltr_model.pkl")
                    except Exception as e:
                        st.error(f"LightGBM not available or failed: {e}")
                else:
                    try:
                        from sklearn.linear_model import LogisticRegression
                        clf = LogisticRegression(max_iter=300)
                        clf.fit(X, y)
                        save_ltr_model(clf)
                        st.success("Trained Logistic Regression ranker and saved to data/models/ltr_model.pkl")
                    except Exception as e:
                        st.error(f"Logistic Regression failed: {e}")

with tabs[2]:
    st.subheader("Feedback Log")
    ensure_feedback_table()
    conn = sqlite3.connect(ARTICLES_DB)
    fb = pd.read_sql_query("SELECT * FROM feedback ORDER BY ts DESC LIMIT 1000", conn)
    conn.close()
    if fb.empty:
        st.info("No feedback yet.")
    else:
        st.dataframe(fb, use_container_width=True, hide_index=True)
        sat = fb['satisfied'].dropna()
        if not sat.empty:
            rate = (sat==1).mean()
            st.write(f"**Satisfaction rate:** {rate:.2%}")
        gold = build_gold_from_feedback()
        if not gold.empty:
            st.download_button(
                "Download gold.csv (optional)",
                data=gold.to_csv(index=False).encode("utf-8"),
                file_name="gold.csv",
                mime="text/csv"
            )

st.caption("© FOBOH – Hybrid retriever with cross-encoder, temporal & sports boosts, feedback evaluation, and optional LTR.")
