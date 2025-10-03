import re, html, string
from urllib.parse import urlparse
from email.utils import parsedate_to_datetime

STOPWORDS = set("""a an and the of to in on for with at by from into under over after before about 
is are was were be been being this that these those it its their his her your our my we you they as 
not or nor but if then else when while do does did doing have has had having out up down off just 
more most less least many much few some any all each every other same own so than too very can 
will would should could might may breaking news update live exclusive analysis
""".split())

def parse_dt(value):
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None

def domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""

def clean_html(s: str|None) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def first_sentence(text: str) -> str:
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    return m[0] if m else text

def lead_summarize(title: str, summary_html: str|None, max_words:int=60) -> str:
    text = clean_html(summary_html) if summary_html else (title or "")
    sent = first_sentence(text)
    words = sent.split()
    return sent if len(words) <= max_words else " ".join(words[:max_words]) + "…"

def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in re.split(r"\W+", t) if w and w not in STOPWORDS and not w.isdigit()]
    return " ".join(tokens)
