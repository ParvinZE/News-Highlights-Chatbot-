# rag_build_index.py — build a vector index over articles or highlights
# Usage:
#   python rag_build_index.py --since 30d --provider local --summary-model google/pegasus-xsum
#   python rag_build_index.py --since 30d --provider local --use-highlights
#   OPENAI_API_KEY=... python rag_build_index.py --since 30d --provider openai --embed-model text-embedding-3-small

import os, re, json, sqlite3, argparse, numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

DATA = Path("data")
DB_DEFAULT = DATA / "articles.db"
IDX_FAISS = DATA / "rag_index.faiss"
IDX_SK    = DATA / "rag_index.sk.pkl"
META_FILE = DATA / "rag_meta.jsonl"
HIGHLIGHTS_JSON = DATA / "highlights.json"

def parse_since(s: str) -> str:
    if re.fullmatch(r"\d+d", s.strip()):
        days = int(s[:-1])
        return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return s

def table_columns(con, table: str) -> List[str]:
    return [name for _, name, *_ in con.execute(f"PRAGMA table_info({table});")]

def build_text_and_model_expr(cols: List[str], summary_model: Optional[str]) -> (str, str, list):
    """
    Returns (text_expr_sql, model_expr_sql, params)
      text_expr_sql  -> yields AS text
      model_expr_sql -> yields AS text_model
      params         -> SQL params to bind (summary_model if provided)
    """
    fallbacks = [c for c in ["summary", "lead", "description", "abstract", "content", "body", "text"] if c in cols]
    fallback_sql = "COALESCE(" + ", ".join(fallbacks) + ", '')" if fallbacks else "''"

    params = []
    if summary_model:
        s_text  = "(SELECT summary FROM abs_summaries WHERE article_id=a.id AND model=? ORDER BY created_at DESC LIMIT 1)"
        s_model = "(SELECT model   FROM abs_summaries WHERE article_id=a.id AND model=? ORDER BY created_at DESC LIMIT 1)"
        params.extend([summary_model, summary_model])
    else:
        s_text  = "(SELECT summary FROM abs_summaries WHERE article_id=a.id ORDER BY created_at DESC LIMIT 1)"
        s_model = "(SELECT model   FROM abs_summaries WHERE article_id=a.id ORDER BY created_at DESC LIMIT 1)"

    text_expr  = f"COALESCE({s_text}, {fallback_sql}) AS text"
    model_expr = f"COALESCE({s_model}, '') AS text_model"
    return text_expr, model_expr, params

def fetch_rows(db_path: Path, since_iso: str, limit: int, summary_model: Optional[str]) -> List[Dict]:
    con = sqlite3.connect(db_path)
    cols = table_columns(con, "articles")
    text_expr, model_expr, params0 = build_text_and_model_expr(cols, summary_model)

    base_cols = ["id","title","url","source_domain","published_at","category","cluster_id","author"]
    sel = [ (f"a.{c}" if c in cols else f"'' AS {c}") for c in base_cols ]
    sel_clause = ", ".join(sel + [text_expr, model_expr])

    q = f"""
    SELECT {sel_clause}
    FROM articles a
    WHERE a.published_at >= ?
    ORDER BY a.published_at DESC
    LIMIT ?;
    """
    params = params0 + [since_iso, limit]
    rows = con.execute(q, params).fetchall()
    con.close()

    keys = base_cols + ["text","text_model"]
    out = []
    for r in rows:
        rec = dict(zip(keys, r))
        out.append({
            "id": rec["id"],
            "title": rec.get("title") or "",
            "text": rec.get("text") or "",
            "text_model": rec.get("text_model") or "",
            "category": rec.get("category") or "",
            "cluster_id": rec.get("cluster_id") or "",
            "author": rec.get("author") or "",
            "url": rec.get("url") or "",
            "source_domain": rec.get("source_domain") or "",
            "published_at": rec.get("published_at") or "",
        })
    return out

def load_highlights(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"highlights.json not found at {path}. Run step3_highlights.py first.")
    raw = json.loads(path.read_text(encoding="utf-8"))

    # highlights may be {"cat":[...]} or a flat list
    items = []
    if isinstance(raw, dict):
        for cat, arr in raw.items():
            for h in arr:
                h = dict(h)
                h.setdefault("category", cat)
                items.append(h)
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("highlights.json format not recognized")

    # Normalize for indexing
    rows = []
    for h in items:
        rid = h.get("top_article_id") or h.get("article_id") or h.get("id") or None
        title = h.get("title","")
        text  = h.get("summary","") or ""
        rows.append({
            "id": rid,
            "title": title,
            "text": text,
            "text_model": h.get("text_model",""),   # may be empty
            "category": h.get("category",""),
            "cluster_id": h.get("cluster_id",""),
            "author": h.get("author",""),
            "url": h.get("url",""),
            "source_domain": h.get("source_domain",""),
            "published_at": h.get("published_at",""),
        })
    return rows

# ---------- Embeddings ----------
def embed_local(model_name: str, texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    X = m.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(X, dtype=np.float32)

def embed_openai(model_name: str, texts: List[str]) -> np.ndarray:
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
    client = OpenAI(api_key=key)
    out = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        resp = client.embeddings.create(model=model_name, input=chunk)
        out.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    X = np.vstack(out)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="30d")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--provider", choices=["local","openai"], default="local")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="local: sbert model; openai: text-embedding-3-small|large")
    ap.add_argument("--db", default=str(DB_DEFAULT), help="Path to SQLite DB")
    ap.add_argument("--summary-model", default=None,
                    help="Prefer abs_summaries from this model when indexing from DB")
    ap.add_argument("--use-highlights", action="store_true",
                    help="Index highlight representatives from data/highlights.json instead of raw articles")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    since_iso = parse_since(args.since)

    if args.use_highlights:
        rows = load_highlights(HIGHLIGHTS_JSON)
        # Optional: filter by 'since' if highlight published_at is present
        if since_iso:
            def _keep(r):
                ts = r.get("published_at") or ""
                try:
                    dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                    return dt >= datetime.fromisoformat(since_iso.replace("Z","+00:00"))
                except Exception:
                    return True
            rows = [r for r in rows if _keep(r)]
    else:
        rows = fetch_rows(Path(args.db), since_iso, args.limit, args.summary_model)

    if not rows:
        print("[rag] no rows found; run ingestion/processing/highlights first.")
        return

    texts = [ (r.get("title","") + " " + r.get("text","")).strip() for r in rows ]
    print(f"[rag] embedding {len(texts)} items with {args.provider}:{args.embed_model} …")

    if args.provider == "local":
        X = embed_local(args.embed_model, texts)
    else:
        X = embed_openai(args.embed_model, texts)

    # ---------- Index ----------
    try:
        import faiss
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)  # normalized vectors => IP ~= cosine
        faiss.write_index(index, str(IDX_FAISS))
        provider = "faiss"
    except Exception as e:
        print(f"[rag] FAISS not available ({e}); using sklearn NearestNeighbors.")
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=50, metric="cosine")
        nn.fit(X)
        import joblib; joblib.dump(nn, IDX_SK)
        provider = "sklearn"

    # ---------- Metadata ----------
    with META_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[rag] wrote index ({provider}) to {IDX_FAISS if provider=='faiss' else IDX_SK} and metadata to {META_FILE}")

if __name__ == "__main__":
    main()
