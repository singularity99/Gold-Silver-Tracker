import feedparser

# Google News RSS feeds -- targeted searches for each catalyst
FEEDS = [
    ("Gold & Silver", "https://news.google.com/rss/search?q=gold+silver+price&hl=en-US&gl=US&ceid=US:en"),
    ("Fed & Rates", "https://news.google.com/rss/search?q=federal+reserve+rate+cut+gold&hl=en-US&gl=US&ceid=US:en"),
    ("Tariffs", "https://news.google.com/rss/search?q=tariffs+gold+silver+trade+war&hl=en-US&gl=US&ceid=US:en"),
    ("Iran & Geopolitics", "https://news.google.com/rss/search?q=iran+conflict+gold+oil&hl=en-US&gl=US&ceid=US:en"),
    ("Central Banks & Asia", "https://news.google.com/rss/search?q=central+bank+gold+buying+china+india&hl=en-US&gl=US&ceid=US:en"),
    ("Silver Industrial", "https://news.google.com/rss/search?q=silver+industrial+demand+solar+EV&hl=en-US&gl=US&ceid=US:en"),
    ("Commodities", "https://www.investing.com/rss/news_285.rss"),
]

CATALYST_KEYWORDS = [
    "gold", "silver", "precious metal", "bullion",
    "fed", "federal reserve", "warsh", "powell", "rate cut", "interest rate",
    "iran", "middle east", "geopolit",
    "tariff", "trade war", "trump",
    "china", "india", "central bank", "reserve",
    "sge", "shanghai gold", "comex",
    "solar", "ev", "industrial demand",
    "etf inflow", "gld", "slv",
    "platinum", "palladium",
]


def fetch_catalyst_news(max_articles: int = 20) -> list[dict]:
    """Fetch and filter news articles matching catalyst keywords from multiple sources."""
    articles = []
    for feed_name, feed_url in FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = (title + " " + summary).lower()
                matched = [kw for kw in CATALYST_KEYWORDS if kw in text]
                if matched:
                    articles.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": feed_name,
                        "matched_keywords": matched,
                    })
        except Exception:
            continue

    # Deduplicate by title
    seen_titles = set()
    unique = []
    for a in articles:
        t = a["title"].lower().strip()
        if t not in seen_titles:
            seen_titles.add(t)
            unique.append(a)

    # Sort by keyword relevance (more keyword matches = more relevant)
    unique.sort(key=lambda x: len(x["matched_keywords"]), reverse=True)
    return unique[:max_articles]
