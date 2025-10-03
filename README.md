# FOBOH – News Highlights & Chatbot

A lightweight, end-to-end pipeline that ingests RSS feeds, deduplicates similar stories, ranks daily highlights with transparent signals, and powers a hybrid (keyword + semantic) RAG chatbot. Everything runs locally on SQLite + Python; no heavy infra required.

---

## ✨ Features

- **Ingestion (Step 1):** Robust RSS fetching via `feedparser` with a `requests` fallback.
- **Processing (Step 2):** Conservative title normalization → clusters + canonical representative.
- **Highlights (Step 3):** Score by distinct sources, cluster size, keyword bonus, and recency (half-life).
- **Summarization (Step 4, optional):** Local seq2seq models (Pegasus/BART) or OpenAI.
- **RAG Index (Step 5):** Hybrid retrieval over highlights with semantic + keyword fusion.
- **UIs:** 
  - `mini_app.py` – category pages with **“Seen on N sources • M related articles”** and score breakdown.
  - `rag_chat_app.py` – ask questions, see similarity scores and metadata.

---

## 🧱 Project Layout

```
.
├─ data/
│  ├─ articles.db           # SQLite (WAL mode)
│  ├─ highlights.json       # Exported by Step 3 (read by mini_app)
│  ├─ rag_index.faiss|sk.pkl# Vector index (FAISS or sklearn)
│  └─ rag_meta.jsonl        # Per-item metadata for RAG
├─ step1_ingest.py
├─ step2_process.py
├─ step3_highlights.py
├─ step4_llm_summarize.py   # optional
├─ rag_build_index.py
├─ mini_app.py              # highlights UI
├─ rag_chat_app.py          # chatbot UI
├─ config.py                # categories, keyword weights, feeds (if present)
└─ requirements.txt
```

---

## 🚀 Quickstart

> **Requires** Python 3.10+ (3.11 recommended).

```bash
# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# 2) Install deps
python -m pip install -r requirements.txt
# (Optional) Try a faster index:
# python -m pip install faiss-cpu || true

# 3) Run the pipeline
python step1_ingest.py --since 2d --limit 300
python step2_process.py --since 365d
python step3_highlights.py --since 365d --topk 5

# 4) Build the RAG index (over highlights)
python rag_build_index.py --since 30d --provider local --use-highlights
# If you prefer to index summaries from DB:
# python rag_build_index.py --since 30d --provider local --summary-model google/pegasus-xsum

# 5) Start the UIs (two terminals or run one at a time)
python -m streamlit run mini_app.py
python -m streamlit run rag_chat_app.py
```

---

## 🔧 Configuration

- **Categories:** `config.py` → `SUPPORTED_CATEGORIES`
- **Keyword weights for scoring:**  
  In `step3_highlights.py` set `KEYWORD_WEIGHTS`, e.g.:
  ```python
  KEYWORD_WEIGHTS = {
    "breaking": 2.0, "exclusive": 1.5, "live": 1.2, "update": 1.0
  }
  ```
- **Recency half-life:** `recency_weight(..., half_life_hours=48.0)`
- **Feeds:** Define your RSS sources in your ingestion script or `config.py`.

---

## 🧠 Summarization

Local models (no API key):
```bash
python step4_llm_summarize.py --since 14d --provider local   --model google/pegasus-xsum --device auto --resummarize
```

OpenAI (set your key first):
```bash
export OPENAI_API_KEY="sk-..."   # replace with your real key
python step4_llm_summarize.py --since 14d --provider openai   --model gpt-4o-mini --resummarize
```

---

## 💬 Chatting

Build (or rebuild) the index:
```bash
python rag_build_index.py --since 30d --provider local --use-highlights
```

Run the app:
```bash
python -m streamlit run rag_chat_app.py
```

Try queries like:
- `RBA outlook this month`
- `Israel ceasefire talks`
- `ASX earnings`
- `AFL finals schedule`

Use the sidebar to **show similarity scores**, **metadata**, and tweak thresholds.

---

## 🧪 Troubleshooting

- **“No RAG index yet”** → rebuild the index:
  ```bash
  python rag_build_index.py --since 30d --provider local --use-highlights
  ```
- **`database is locked`** → close Streamlit tabs and **pause OneDrive/Dropbox** while steps run.
- **FAISS install fails** → skip it; the code falls back to `scikit-learn` automatically.
- **Nothing shows in a category** → ensure feeds populated `source_domain` and rerun:
  ```bash
  python step2_process.py --since 365d
  python step3_highlights.py --since 365d --topk 5
  ```
- **`streamlit: command not found`** → `pip install streamlit` inside your venv.
- **MPS/CUDA issues** → force device for local summarization:
  ```bash
  python step4_llm_summarize.py ... --device cpu
  ```

---

## 🔐 Env Vars

- `OPENAI_API_KEY` (optional) — only needed if you choose `--provider openai`.

---

## 📄 License

Add your preferred license (MIT/Apache-2.0) here.

---

## 🤝 Contributing

Issues and PRs welcome. Please include:
- steps to reproduce,
- OS/Python version,
- logs from the failing script.
