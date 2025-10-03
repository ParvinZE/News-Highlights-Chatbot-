# A curated set of reliable feeds
# ABC "Just In" and ABC Business/Sport are very active.
# Reuters Top News and Business are typically accessible.
SOURCES = {
    "general": [
        "https://www.abc.net.au/news/feed/51120/rss.xml",  # ABC Just In
        "http://feeds.reuters.com/reuters/topNews",        # Reuters Top
    ],
    "finance": [
        "https://www.abc.net.au/news/feed/51620/rss.xml",  # ABC Business
        "http://feeds.reuters.com/reuters/businessNews",    # Reuters Business
    ],
    "sports": [
        "https://www.abc.net.au/news/feed/45910/rss.xml",  # ABC Sport
    ],
    "entertainment": [
        "https://www.abc.net.au/news/feed/45920/rss.xml",  # ABC Entertainment
    ],
}
SUPPORTED_CATEGORIES = list(SOURCES.keys())
KEYWORDS = ["breaking","exclusive","live","developing","update","analysis"]
