import feedparser
from datetime import datetime, timedelta

FEEDS = [
    "https://www.kitco.com/rss/kitco.xml",
    "https://feeds.reuters.com/reuters/businessNews",
]

CATALYST_KEYWORDS = [
    "gold", "silver", "precious metal",
    "fed", "federal reserve", "warsh", "powell", "rate cut", "interest rate",
    "iran", "middle east", "geopolit",
    "tariff", "trade war", "trump",
    "china gold", "india gold", "central bank", "reserve",
    "sge", "shanghai gold", "comex",
    "solar", "ev", "industrial demand",
    "etf inflow", "gld", "slv",
]


def fetch_catalyst_news(max_articles: int = 10) -> list[dict]:
    """Fetch and filter news articles matching catalyst keywords."""
    articles = []
    for feed_url in FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "").lower()
                summary = entry.get("summary", "").lower()
                text = title + " " + summary
                matched = [kw for kw in CATALYST_KEYWORDS if kw in text]
                if matched:
                    articles.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": feed_url.split("/")[2],
                        "matched_keywords": matched,
                    })
        except Exception:
            continue

    seen_titles = set()
    unique = []
    for a in articles:
        t = a["title"].lower().strip()
        if t not in seen_titles:
            seen_titles.add(t)
            unique.append(a)

    unique.sort(key=lambda x: len(x["matched_keywords"]), reverse=True)
    return unique[:max_articles]
