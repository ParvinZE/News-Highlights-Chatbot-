# rag_chat_app.py — FOBOH RAG chat with scores & rich metadata
# Run:
#   python -m streamlit run rag_chat_app.py
#
# Requirements:
#   pip install streamlit numpy sentence-transformers
#   pip install faiss-cpu           # optional, recommended
#   pip install openai              # if you enable OpenAI generation
#
# Index files are created by rag_build_index.py:
#   data/rag_index.faiss  (or data/rag_index.sk.pkl if using sklearn fallback)
#   data/rag_meta.jsonl   (one JSON per line; includes text_model if you used latest builder)

# --- Inline, cloud-friendly RAG index loader/builder ---
from pathlib import Path
import os, json, pickle
import streamlit as st

DATA_DIR = Path("data")
SK_PATH   = DATA_DIR / "rag_index.sk.pkl"
META_PATH = DATA_DIR / "rag_meta.jsonl"
HIGHLIGHTS_JSON = DATA_DIR / "highlights.json"

@st.cache_resource
def get_embedder():
    # small, CPU-friendly embedding model
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _load_sklearn_index():
    if SK_PATH.exists() and META_PATH.exists():
        with open(SK_PATH, "rb") as f:
            nn = pickle.load(f)
        with META_PATH.open("r", encoding="utf-8") as f:
            meta = [json.loads(line) for line in f]
        return nn, meta
    return None, None

def _save_sklearn_index(nn, meta):
    with open(SK_PATH, "wb") as f:
        pickle.dump(nn, f)
    with META_PATH.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def build_index_inline_from_highlights():
    """Build a sklearn NearestNeighbors index from data/highlights.json (no DB, no subprocess)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HIGHLIGHTS_JSON.exists():
        st.error("data/highlights.json not found. Run step3_highlights.py locally and commit it, or generate it in the cloud.")
        st.stop()

    data = json.loads(HIGHLIGHTS_JSON.read_text(encoding="utf-8"))
    records = []  # (text, meta)
    for cat, items in (data or {}).items():
        for it in items:
            text = f"{it.get('title','')} {it.get('summary','')}".strip()
            if not text:
                continue
            meta = {
                "category": cat,
                "title": it.get("title",""),
                "summary": it.get("summary",""),
                "url": it.get("url",""),
                "score": it.get("score_total") or it.get("score") or 0.0,
                "freq_sources": it.get("freq_sources"),
                "cluster_size": it.get("cluster_size"),
                "recency_weight": it.get("recency_weight"),
            }
            records.append((text, meta))

    if not records:
        st.error("highlights.json has no items. Re-run step3_highlights.py to populate it.")
        st.stop()

    # Embed
    embedder = get_embedder()
    texts = [t for t, _ in records]
    X = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # sklearn index (cosine via metric='cosine')
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="cosine")
    nn.fit(X)

    meta = [m for _, m in records]
    _save_sklearn_index(nn, meta)
    return nn, meta

def load_or_build_index():
    # Prefer sklearn files if present (portable on Streamlit Cloud)
    nn, meta = _load_sklearn_index()
    if nn is not None:
        return "sklearn", nn, meta

    # If no index, offer to build inline (no subprocess)
    st.warning("No RAG index yet.")
    if st.button("Build index now"):
        with st.spinner("Building sklearn index from highlights…"):
            nn, meta = build_index_inline_from_highlights()
        st.success("Index built. Reloading…")
        st.rerun()
    st.stop()

# Call this once near the top of your app
BACKEND, INDEX, META = load_or_build_index()

def search_query(query: str, topk: int = 5):
    import numpy as np
    embedder = get_embedder()
    qv = embedder.encode([query], normalize_embeddings=True)
    # sklearn NearestNeighbors returns distances (cosine distance),
    # so similarity = 1 - distance
    distances, indices = INDEX.kneighbors(qv, n_neighbors=min(topk, len(META)))
    hits = []
    for rank, (d, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        m = META[idx]
        sim = float(1.0 - d)
        hits.append({"rank": rank, "similarity": sim, **m})
    return hits

if backend is None:
    st.warning("No RAG index yet.")
    if st.button("Build index now"):
        with st.spinner("Building sklearn index from highlights…"):
            # Build over highlights so we don't need the DB in the cloud
            subprocess.run(["python", "rag_build_index.py", "--since", "30d", "--provider", "local", "--use-highlights"], check=True)
        st.success("Index built. Reloading…")
        st.rerun()
    st.stop()

st.set_page_config(page_title="FOBOH – RAG Chat", layout="wide")
st.title("FOBOH – RAG Chat over Articles")
st.caption("Retrieval-augmented answers with similarity scores and metadata.")

# ---------------- Sidebar controls ----------------
data_dir = st.sidebar.text_input("Data directory", value="data")
provider = st.sidebar.selectbox("Embedding provider", ["local", "openai"], index=0)
embed_model = st.sidebar.text_input(
    "Embedding model",
    "sentence-transformers/all-MiniLM-L6-v2" if provider=="local" else "text-embedding-3-small"
)
gen_provider = st.sidebar.selectbox("Generator (LLM for answers)", ["openai", "none"], index=1)
gen_model = st.sidebar.text_input("Generation model (if OpenAI)", "gpt-4o-mini")
topk = st.sidebar.slider("Top-K passages", 3, 15, 6, 1)
show_scores = st.sidebar.checkbox("Show similarity scores", value=True)
show_meta   = st.sidebar.checkbox("Show metadata under each hit", value=True)

DATA = Path(data_dir)
IDX_FAISS = DATA / "rag_index.faiss"
IDX_SK    = DATA / "rag_index.sk.pkl"
META_FILE = DATA / "rag_meta.jsonl"

# ---------------- Loaders ----------------
@st.cache_resource(show_spinner=False)
def load_index(idx_faiss: Path, idx_sk: Path):
    if idx_faiss.exists():
        try:
            import faiss
            return ("faiss", faiss.read_index(str(idx_faiss)))
        except Exception as e:
            st.warning(f"FAISS index present but could not load ({e}). Will try sklearn fallback…")
    if idx_sk.exists():
        try:
            import joblib
            nn = joblib.load(idx_sk)
            return ("sk", nn)
        except Exception as e:
            st.error(f"Could not load sklearn index: {e}")
            return None
    return None

@st.cache_data(show_spinner=False)
def load_meta(meta_file: Path):
    if not meta_file.exists():
        return []
    return [json.loads(line) for line in meta_file.open("r", encoding="utf-8")]

index = load_index(IDX_FAISS, IDX_SK)
meta  = load_meta(META_FILE)

# ---------------- Embedding helpers ----------------
def embed_query_local(model_name: str, text: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    v = m.encode([text], normalize_embeddings=True)
    return v.astype("float32")

def embed_query_openai(model_name: str, text: str) -> np.ndarray:
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=key)
    r = client.embeddings.create(model=model_name, input=[text])
    v = np.asarray(r.data[0].embedding, dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v[None, :]

def answer_openai(model: str, question: str, contexts: list) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    citations = "\n\n".join([f"[{i+1}] {c['title']} — {c['url']}" for i, c in enumerate(contexts)])
    ctx = "\n\n---\n\n".join([f"({i+1}) {c['title']}\n{c['text']}" for i, c in enumerate(contexts)])
    prompt = (
        "Answer the user using ONLY the context chunks. If unsure, say you don't know.\n"
        "Cite sources inline like [1], [2]. Keep the answer concise and factual.\n\n"
        f"Context:\n{ctx}\n\nUser question: {question}\n"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a careful news assistant that cites sources."},
            {"role":"user","content":prompt}
        ],
        temperature=0.2,
        max_tokens=400,
    )
    out = r.choices[0].message.content.strip()
    out += "\n\nSources:\n" + citations
    return out

# ---------------- Search over the index ----------------
def search(query_vec: np.ndarray, topk: int):
    if index is None: return []
    kind, obj = index
    if kind == "faiss":
        D, I = obj.search(query_vec.astype("float32"), topk)
        return list(zip(I[0].tolist(), D[0].tolist()))
    else:
        # sklearn NearestNeighbors fitted on X (cosine metric)
        sims, idxs = obj.kneighbors(query_vec, n_neighbors=topk, return_distance=True)
        sims = 1.0 - sims[0]  # cosine distance -> similarity
        return list(zip(idxs[0].tolist(), sims.tolist()))

# ---------------- Empty-state guidance ----------------
if index is None or not meta:
    st.warning(
        "No RAG index yet. Build it in a terminal:\n\n"
        "`python rag_build_index.py --since 30d --provider local --summary-model google/pegasus-xsum`\n\n"
        "Then refresh this page."
    )

# Still show the chat input so users know where to type
q = st.chat_input("Ask about the news (e.g., 'Israel latest', 'RBA outlook', 'AFL finals')")
if not q:
    st.stop()

with st.chat_message("user"):
    st.markdown(q)

with st.chat_message("assistant"):
    if index is None or not meta:
        st.info("Index not loaded yet. Build the index, then ask again.")
        st.stop()

    # 1) Embed the query
    try:
        qv = embed_query_local(embed_model, q) if provider=="local" else embed_query_openai(embed_model, q)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        st.stop()

    # 2) Retrieve
    hits = search(qv, topk=topk)
    if not hits:
        st.markdown("No results in the index. Try rebuilding with a wider window.")
        st.stop()

    # 3) Render retrieved items with scores and metadata
    ctx = []
    for i, (idx, sim) in enumerate(hits, start=1):
        if idx < 0 or idx >= len(meta):  # safety
            continue
        mrec = meta[idx]
        title = mrec.get("title") or "(untitled)"
        url   = mrec.get("url") or ""
        src   = mrec.get("source_domain") or ""
        cat   = mrec.get("category") or ""
        when  = mrec.get("published_at") or ""
        cid   = mrec.get("cluster_id") or ""
        tmodel= mrec.get("text_model") or ""   # set by new rag_build_index.py

        header = f"**[{i}] {title}**"
        if show_scores:
            header += f" — sim {sim:.2f}"
        st.markdown(header)
        if url:
            st.write(url)

        if show_meta:
            with st.expander("Details", expanded=False):
                st.markdown(
                    f"- **Source:** {src or '—'}  \n"
                    f"- **Published:** {when or '—'}  \n"
                    f"- **Category:** {cat or '—'}  \n"
                    f"- **Cluster ID:** {cid or '—'}  \n"
                    f"- **Summary model:** {tmodel or '—'}"
                )

        s = (mrec.get("text") or "").strip()
        ctx.append({"title": title, "text": s, "url": url})

    st.markdown("---")

    # 4) Generate an answer (optional)
    if gen_provider == "openai":
        try:
            ans = answer_openai(gen_model, q, ctx)
            st.markdown(ans)
        except Exception as e:
            st.error(f"LLM error: {e}")
            st.info("Retrieval worked; to enable generation, set OPENAI_API_KEY and check your model name.")
    else:
        st.info("Retrieval-only mode. Enable OpenAI in the sidebar to generate an answer with citations.")
