import argparse, json, feedparser, requests
from datetime import datetime, timedelta, timezone
from config import SOURCES, SUPPORTED_CATEGORIES
from db import init_db, get_conn
from util import parse_dt, domain

UA_HDRS = {"User-Agent": "Mozilla/5.0 (FOBOH/1.0; +https://example.com)"}

def parse_since(value: str|None):
    if not value:
        return datetime.now(timezone.utc) - timedelta(days=30)
    import re
    m = re.fullmatch(r"(\d+)([dhmy])", value.strip())
    if m:
        n, u = int(m.group(1)), m.group(2)
        if u == "h": delta = timedelta(hours=n)
        elif u == "d": delta = timedelta(days=n)
        elif u == "m": delta = timedelta(days=30*n)
        else: delta = timedelta(days=365*n)
        return datetime.now(timezone.utc) - delta
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=30)

def fetch_with_fallback(url: str):
    # 1) normal feedparser (urllib)
    f = feedparser.parse(url, request_headers=UA_HDRS)
    status = getattr(f, "status", None)
    if getattr(f, "entries", None) and not getattr(f, "bozo", False):
        return f, {"mode": "urllib", "status": status}
    # 2) requests fallback
    try:
        r = requests.get(url, headers=UA_HDRS, timeout=20)
        r.raise_for_status()
        f2 = feedparser.parse(r.content)
        return f2, {"mode": "requests", "status": r.status_code}
    except Exception as e:
        # return the original f so we can inspect bozo_exception
        setattr(f, "fallback_exception", e)
        return f, {"mode": "urllib->requests_failed", "status": status}

def ingest(categories, since_dt, limit, use_sample=False):
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    total_inserted = 0

    if use_sample:
        from pathlib import Path
        data = json.loads((Path(__file__).resolve().parent / "sample_data" / "sample_rss_dump.json").read_text())
        for item in data:
            title = item.get("title"); url = item.get("url"); cat = item.get("category_feed","general")
            if not title or not url: continue
            dt_pub = parse_dt(item.get("published")) or datetime.now(timezone.utc)
            cur.execute("""
                INSERT OR IGNORE INTO articles
                (title, author, source, source_domain, url, category_feed, summary_raw, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (title, item.get("author"), item.get("source","Sample"), domain(url), url, cat,
                  item.get("summary_raw"), dt_pub.isoformat(), datetime.now(timezone.utc).isoformat()))
            total_inserted += cur.rowcount
        conn.commit(); conn.close()
        print(f"[step1] inserted {total_inserted} new articles from sample.")
        return

    for cat in categories:
        for feed_url in SOURCES[cat]:
            feed, meta = fetch_with_fallback(feed_url)
            status = meta.get("status")
            mode = meta.get("mode")
            entries = getattr(feed, "entries", []) or []
            print(f"[step1] Feed {cat} -> {feed_url} mode={mode} status={status} bozo={getattr(feed,'bozo',None)} items={len(entries)}")
            if getattr(feed, "bozo", False):
                print("  bozo_exception:", getattr(feed, "bozo_exception", None))
                if hasattr(feed, "fallback_exception"):
                    print("  fallback_exception:", getattr(feed, "fallback_exception"))
            source = getattr(feed, "feed", {}).get("title", feed_url) if getattr(feed, "feed", None) else feed_url
            for e in entries[:limit]:
                url = e.get("link") or None
                title = e.get("title") or None
                if not url or not title:
                    continue
                published = e.get("published") or e.get("updated") or None
                dt_pub = parse_dt(published) or datetime.now(timezone.utc)
                if dt_pub.tzinfo is None:
                    dt_pub = dt_pub.replace(tzinfo=timezone.utc)
                if since_dt and dt_pub < since_dt:
                    continue
                author = e.get("author") or e.get("dc_creator") or None
                summary_raw = e.get("summary") or None
                cur.execute("""
                    INSERT OR IGNORE INTO articles
                    (title, author, source, source_domain, url, category_feed, summary_raw, published_at, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    title.strip(), author, source.strip(), domain(url), url, cat,
                    summary_raw, dt_pub.isoformat(), datetime.now(timezone.utc).isoformat()
                ))
                total_inserted += cur.rowcount
    conn.commit(); conn.close()
    print(f"[step1] inserted {total_inserted} new articles.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='*', default=SUPPORTED_CATEGORIES)
    ap.add_argument('--since', type=str, default='30d')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--use-sample', action='store_true', help='Ingest bundled sample data instead of RSS (works offline)')
    args = ap.parse_args()
    since_dt = parse_since(args.since)
    cats = [c for c in args.categories if c in SUPPORTED_CATEGORIES]
    ingest(cats, since_dt, args.limit, use_sample=args.use_sample)
