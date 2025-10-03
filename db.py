import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "articles.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = '''
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT,
    source TEXT NOT NULL,
    source_domain TEXT,
    url TEXT NOT NULL,
    category_feed TEXT NOT NULL,
    category_final TEXT,
    summary_raw TEXT,
    summary_abs TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    norm_title TEXT,
    cluster_id INTEGER,
    dedup_canonical INTEGER DEFAULT 0,
    UNIQUE(url, published_at)
);
CREATE INDEX IF NOT EXISTS idx_cat_pub ON articles(category_final, published_at);
CREATE INDEX IF NOT EXISTS idx_norm_title ON articles(norm_title);
CREATE INDEX IF NOT EXISTS idx_cluster ON articles(cluster_id);

CREATE TABLE IF NOT EXISTS highlights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    top_article_id INTEGER NOT NULL,
    score REAL NOT NULL,
    freq_sources INTEGER,
    cluster_size INTEGER,
    keywords INTEGER,
    created_at TEXT NOT NULL,          -- YYYY-MM-DD
    UNIQUE(category, cluster_id, created_at)
);
'''

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = get_conn()
    try:
        for stmt in SCHEMA.split(';'):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
    finally:
        conn.close()
