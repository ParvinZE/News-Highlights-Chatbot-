# step4_llm_summarize.py — robust LLM summarization into SQLite
# Works with OpenAI (set OPENAI_API_KEY) or local HF models (e.g., facebook/bart-large-cnn).
#
# Examples:
#   OPENAI_API_KEY=sk-... python step4_llm_summarize.py --since 14d --provider openai --model gpt-4o-mini
#   python step4_llm_summarize.py --since 14d --provider local  --model facebook/bart-large-cnn
#
# Local deps:
#   python -m pip install transformers sentencepiece torch

import os, re, sqlite3, argparse, time, warnings
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone, timedelta

# Quiet Transformers/tokenizers logs
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

DB = Path("data/articles.db")

# ----------------------- Device + cached pipeline -----------------------
from functools import lru_cache
from transformers import pipeline, AutoConfig

FORCED_DEVICE = None  # set in main() before any summarization

def _pick_device() -> int:
    """Return transformers device index: 0 for GPU/MPS, -1 for CPU."""
    try:
        import torch
    except Exception:
        return -1
    # explicit overrides
    if FORCED_DEVICE == "cpu":  return -1
    if FORCED_DEVICE == "cuda": return 0 if torch.cuda.is_available() else -1
    if FORCED_DEVICE == "mps":  return 0 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else -1
    # auto
    if torch.cuda.is_available(): return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return 0
    return -1

@lru_cache(maxsize=4)
def get_summarizer(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name)
    if not getattr(cfg, "is_encoder_decoder", False):
        raise ValueError(
            f"Model '{model_name}' is not an encoder-decoder summarizer.\n"
            "Use: 'facebook/bart-large-cnn', 'google/pegasus-xsum', "
            "'google/pegasus-cnn_dailymail', or 't5-small'."
        )
    dev = _pick_device()
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=dev)

def summarize_local(model_name: str, title: str, text: str) -> str:
    """Cached pipeline + auto-length; skip model for very short inputs."""
    import re as _re
    summarizer = get_summarizer(model_name)
    tok = summarizer.tokenizer

    chunk = (title + ". " + (text or "")).strip()
    if not chunk:
        return ""

    enc = tok(chunk, truncation=True, max_length=1024, return_tensors=None)
    n_tok = len(enc["input_ids"])

    if n_tok < 40:
        sents = _re.split(r'(?<=[.!?])\s+', chunk)
        return " ".join(sents[:2]).strip()[:400]

    max_len = max(50, min(180, int(0.5 * n_tok)))
    min_len = max(30, min(max_len - 10, int(0.25 * n_tok)))

    out = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        clean_up_tokenization_spaces=True,
    )
    return (out[0]["summary_text"] or "").strip()

# ----------------------- DB helpers -----------------------

def connect(db_path: Path = DB):
    # autocommit + WAL + busy timeout for fewer lock issues
    con = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=60000;")
    con.execute("PRAGMA wal_autocheckpoint=1000;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS abs_summaries (
          article_id INTEGER NOT NULL,
          model      TEXT    NOT NULL,
          summary    TEXT    NOT NULL,
          created_at TEXT    NOT NULL,
          PRIMARY KEY(article_id, model)
        )
    """)
    return con

def parse_since(s: str) -> str:
    if re.fullmatch(r"\d+d", s):
        days = int(s[:-1])
        return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return s

def table_columns(con, table: str) -> List[str]:
    return [name for _, name, *_ in con.execute(f"PRAGMA table_info({table});")]

def pick_text_expr(cols: List[str]) -> str:
    candidates = ["summary", "lead", "description", "abstract", "content", "body", "text"]
    present = [c for c in candidates if c in cols]
    return "COALESCE(" + ", ".join(present) + ", '')" if present else "''"

def load_candidates(con, since_iso: str, limit: int) -> List[Dict]:
    cols = table_columns(con, "articles")
    text_expr = pick_text_expr(cols) + " AS text"
    base_cols = ["id", "title", "url", "source_domain", "published_at"]
    sel = [c if c in cols else f"'' AS {c}" for c in base_cols]
    sel_clause = ", ".join(sel + [text_expr])

    q = f"""
    SELECT {sel_clause}
    FROM articles
    WHERE published_at >= ?
    ORDER BY published_at DESC
    LIMIT ?;
    """
    rows = con.execute(q, (since_iso, limit)).fetchall()
    keys = [x.split(" AS ")[-1] if " AS " in x else x for x in (base_cols + ["text"])]
    out = []
    for r in rows:
        rec = dict(zip(keys, r))
        out.append({
            "article_id": rec.get("id"),
            "title": rec.get("title") or "",
            "text": rec.get("text") or "",
            "url": rec.get("url") or "",
            "domain": rec.get("source_domain") or "",
            "published_at": rec.get("published_at") or "",
        })
    return out

# ----------------------- Providers -----------------------

def summarize_openai(model: str, title: str, text: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install OpenAI client: pip install openai>=1.0.0") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")
    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a news editor. Write a crisp, factual 2–3 sentence summary.\n"
        "Do not invent facts. Prefer what is in the input. Keep proper nouns.\n\n"
        f"Title: {title}\n"
        f"Text: {text[:6000]}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You write concise factual news summaries."},
            {"role":"user","content":prompt}
        ],
        temperature=0.2,
        max_tokens=220,
    )
    return (resp.choices[0].message.content or "").strip()

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="14d", help="window, e.g. 14d or ISO timestamp")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--provider", choices=["openai", "local"], default="openai")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model or HF model id")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="Force device for local models; 'auto' picks CUDA>MPS>CPU")
    ap.add_argument("--resummarize", action="store_true",
                    help="Overwrite existing summaries for this model (article_id, model)")
    ap.add_argument("--db", default=str(DB), help="Path to SQLite DB (use to avoid cloud-sync locks)")
    args = ap.parse_args()

    # set device BEFORE any summarizer is created
    globals()["FORCED_DEVICE"] = args.device

    since_iso = parse_since(args.since)
    con = connect(Path(args.db))
    rows = load_candidates(con, since_iso, args.limit)
    print(f"[step4] found {len(rows)} recent articles since {since_iso}")

    cur = con.cursor()
    inserted = skipped_existing = errors = batch = 0

    for r in rows:
        exists = cur.execute(
            "SELECT 1 FROM abs_summaries WHERE article_id=? AND model=?",
            (r["article_id"], args.model)
        ).fetchone()
        if exists and not args.resummarize:
            skipped_existing += 1
            continue

        try:
            if args.provider == "openai":
                s = summarize_openai(args.model, r["title"], r["text"])
            else:
                s = summarize_local(args.model, r["title"], r["text"])
        except Exception as e:
            print(f"[step4] skip id={r['article_id']} ({e})")
            errors += 1
            continue

        # retry insert if DB is briefly busy/locked
        for attempt in range(6):
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO abs_summaries(article_id, model, summary, created_at) "
                    "VALUES(?,?,?,datetime('now'))",
                    (r["article_id"], args.model, s)
                )
                break
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

        inserted += 1
        batch += 1
        if batch % 10 == 0:
            con.commit()

    con.commit(); con.close()
    print(f"[step4] inserted={inserted} skipped_existing={skipped_existing} errors={errors}")

if __name__ == "__main__":
    main()
