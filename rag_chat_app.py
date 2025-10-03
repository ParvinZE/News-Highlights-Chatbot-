# rag_chat_app.py — FOBOH RAG chat with scores & rich metadata
# Run:  python -m streamlit run rag_chat_app.py
#
# Requirements:
#   pip install streamlit numpy scikit-learn sentence-transformers joblib
#   (optional) pip install faiss-cpu
#   (optional, for generation) pip install openai

from pathlib import Path
import os, json
import joblib
import numpy as np
import streamlit as st

# ---------------- Page config (must be first) ----------------
st.set_page_config(page_title="FOBOH – RAG Chat", layout="wide")
st.title("FOBOH – RAG Chat over Highlights")
st.caption("Retrieval-augmented answers with similarity scores and metadata.")

# ---------------- Sidebar controls ----------------
data_dir = Path(st.sidebar.text_input("Data directory", value="data"))
embed_provider = st.sidebar.selectbox("Embedding provider", ["local", "openai"], index=0)
embed_model = st.sidebar.text_input(
    "Embedding model",
    "sentence-transformers/all-MiniLM-L6-v2" if embed_provider=="local" else "text-embedding-3-small"
)
gen_provider = st.sidebar.selectbox("Generator (LLM for answers)", ["none", "openai"], index=0)
gen_model = st.sidebar.text_input("Generation model (if OpenAI)", "gpt-4o-mini")
topk = st.sidebar.slider("Top-K passages", 3, 15, 6, 1)
show_scores = st.sidebar.checkbox("Show similarity scores", value=True)
show_meta   = st.sidebar.checkbox("Show metadata under each hit", value=True)

# Paths (based on sidebar)
IDX_FAISS = data_dir / "rag_index.faiss"
IDX_SK    = data_dir / "rag_index.sk.joblib"     # unified: joblib
META_FILE = data_dir / "rag_meta.jsonl"
HIGHLIGHTS_JSON = data_dir / "highlights.json"

# ---------------- Embedding model ----------------
@st.cache_resource(show_spinner=False)
def get_local_embedder(model_name: str):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def embed_query_local(model_name: str, text: str) -> np.ndarray:
    m = get_local_embedder(model_name)
    v = m.encode([text], normalize_embeddings=True, show_progress_bar=False)
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

# ---------------- Index I/O ----------------
def load_meta(meta_file: Path):
    if not meta_file.exists(): return []
    with meta_file.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_meta(meta_file: Path, meta: list):
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with meta_file.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def load_index(idx_faiss: Path, idx_sk: Path):
    # Prefer FAISS if present (created by external builder)
    if idx_faiss.exists():
        try:
            import faiss
            return ("faiss", faiss.read_index(str(idx_faiss)))
        except Exception as e:
            st.warning(f"FAISS index present but could not load ({e}). Using sklearn if available…")
    if idx_sk.exists():
        try:
            nn = joblib.load(idx_sk)
            return ("sk", nn)
        except Exception as e:
            st.error(f"Could not load sklearn index: {e}")
            return None
    return None

def save_sklearn_index(idx_sk: Path, nn):
    idx_sk.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(nn, idx_sk)

# ---------------- Inline builder from highlights ----------------
def build_index_from_highlights(highlights_json: Path):
    if not highlights_json.exists():
        st.error(f"{highlights_json} not found. Run step3_highlights.py to generate it.")
        st.stop()

    data = json.loads(highlights_json.read_text(encoding="utf-8"))
    records = []
    for cat, items in (data or {}).items():
        for it in items:
            title = (it.get("title") or "").strip()
            summ  = (it.get("summary") or "").strip()
            text  = f"{title}. {summ}".strip(". ").strip()
            if not text:
                continue
            meta = {
                "category": cat,
                "title": title,
                "summary": summ,
                "text": text,                      # <-- ensure text is available for generation
                "url": it.get("url",""),
                "score": it.get("score_total") or it.get("score") or 0.0,
                "freq_sources": it.get("freq_sources"),
                "cluster_size": it.get("cluster_size"),
                "recency_weight": it.get("recency_weight"),
                # optional fields if your exporter added them:
                "source_domain": it.get("source_domain"),
                "published_at": it.get("published_at"),
                "cluster_id": it.get("cluster_id"),
                "text_model": it.get("text_model"),
            }
            records.append((text, meta))

    if not records:
        st.error("highlights.json has no items. Re-run step3_highlights.py to populate it.")
        st.stop()

    # Embed all texts
    embedder = get_local_embedder("sentence-transformers/all-MiniLM-L6-v2")
    texts = [t for t, _ in records]
    X = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")

    # sklearn NearestNeighbors on cosine distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="cosine").fit(X)

    meta = [m for _, m in records]
    return nn, meta

# ---------------- Ensure index exists (load or build) ----------------
index = load_index(IDX_FAISS, IDX_SK)
meta = load_meta(META_FILE)

if index is None or not meta:
    st.warning("No RAG index found. Building a sklearn index from highlights.json…")
    with st.spinner("Building index…"):
        nn, meta = build_index_from_highlights(HIGHLIGHTS_JSON)
        save_sklearn_index(IDX_SK, nn)
        save_meta(META_FILE, meta)
        index = ("sk", nn)
    st.success("Index built.")

# ---------------- Search ----------------
def search(index_tuple, query_vec: np.ndarray, topk: int):
    kind, obj = index_tuple
    if kind == "faiss":
        D, I = obj.search(query_vec.astype("float32"), min(topk, len(meta)))
        # If vectors are normalized and FAISS uses inner-product, similarity = dot = 1 - distance (if L2 on unit vecs)
        # Here we assume IP; if your FAISS index is L2 on normalized vectors, also okay to treat 1-D as similarity proxy.
        sims = D[0].tolist()
        idxs = I[0].tolist()
        return list(zip(idxs, sims))
    else:
        # sklearn NearestNeighbors returns cosine distance; similarity = 1 - distance
        distances, indices = obj.kneighbors(query_vec, n_neighbors=min(topk, len(meta)))
        sims = (1.0 - distances[0]).tolist()
        idxs = indices[0].tolist()
        return list(zip(idxs, sims))

# ---------------- Optional generator ----------------
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

# ---------------- Chat UI ----------------
q = st.chat_input("Ask about the news (e.g., 'Israel latest', 'RBA outlook', 'AFL finals')")
if not q:
    st.stop()

with st.chat_message("user"):
    st.markdown(q)

with st.chat_message("assistant"):
    # 1) Embed the query
    try:
        qv = embed_query_local(embed_model, q) if embed_provider=="local" else embed_query_openai(embed_model, q)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        st.stop()

    # 2) Retrieve
    hits = search(index, qv, topk=topk)
    if not hits:
        st.markdown("No results in the index. Try rebuilding with a wider window in step3_highlights.py.")
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
        tmodel= mrec.get("text_model") or ""

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

        text = (mrec.get("text") or f"{mrec.get('title','')} {mrec.get('summary','')}").strip()
        ctx.append({"title": title, "text": text, "url": url})

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
