import argparse
from datetime import datetime, timedelta, timezone
from db import get_conn, init_db
from util import normalize_title, lead_summarize

def parse_since(value: str|None):
    if not value:
        return None
    import re
    m = re.fullmatch(r"(\d+)([dhmy])", value.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == 'h': delta = timedelta(hours=n)
        elif unit == 'd': delta = timedelta(days=n)
        elif unit == 'm': delta = timedelta(days=30*n)
        else: delta = timedelta(days=365*n)
        return (datetime.now(timezone.utc) - delta).isoformat()
    return value

def recompute_clusters(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM articles ORDER BY (published_at IS NULL), published_at DESC, id DESC")
    rows = cur.fetchall()
    by_norm = {}
    for (aid, title) in rows:
        norm = normalize_title(title or "")
        by_norm.setdefault(norm, []).append(aid)
    cluster_id = 1
    for norm, ids in by_norm.items():
        cur.execute("UPDATE articles SET norm_title=? WHERE id IN (%s)" % ",".join("?"*len(ids)), (norm, *ids))
        if len(ids) == 1:
            cur.execute("UPDATE articles SET cluster_id=? WHERE id=?", (1000000+ids[0], ids[0]))
            cur.execute("UPDATE articles SET dedup_canonical=1 WHERE id=?", (ids[0],))
        else:
            for aid in ids:
                cur.execute("UPDATE articles SET cluster_id=? WHERE id=?", (cluster_id, aid))
            can_id = max(ids)
            cur.execute("UPDATE articles SET dedup_canonical=0 WHERE cluster_id=?", (cluster_id,))
            cur.execute("UPDATE articles SET dedup_canonical=1 WHERE id=?", (can_id,))
            cluster_id += 1
    conn.commit()

def process(since_iso: str|None):
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    q = "SELECT id, title, summary_raw, category_feed FROM articles"
    params = []
    if since_iso:
        q += " WHERE (published_at IS NULL OR published_at >= ?)"
        params.append(since_iso)
    cur.execute(q, params); rows = cur.fetchall()
    for (aid, title, summary_raw, cat_feed) in rows:
        summary_abs = lead_summarize(title or "", summary_raw, max_words=60)
        cur.execute("UPDATE articles SET category_final=?, summary_abs=? WHERE id=?", (cat_feed, summary_abs, aid))
    conn.commit()
    recompute_clusters(conn)
    conn.close()
    print(f"[step2] processed {len(rows)} articles and recomputed clusters.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', type=str, default='365d')
    args = ap.parse_args()
    process(parse_since(args.since))
