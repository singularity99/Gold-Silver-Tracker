import feedparser
import re
import html
import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor, as_completed


def _parse_date(date_str: str) -> float:
    """Parse RSS date string to Unix timestamp for sorting. Returns 0 on failure."""
    if not date_str:
        return 0.0
    try:
        return parsedate_to_datetime(date_str).timestamp()
    except Exception:
        pass
    try:
        return datetime.fromisoformat(date_str).timestamp()
    except Exception:
        return 0.0

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

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

_BOILERPLATE_RE = re.compile(
    r"cookie|subscribe|sign.?up|newsletter|javascript|read more|"
    r"advertis|copyright|\ball rights\b|privacy.?policy|terms of|"
    r"log.?in|create.?account|download.?(the|our|app)|also read|"
    r"related.?article|recommended|trending|share.?this|follow.?us|"
    r"skip.?to|breadcrumb|sidebar|@font|font-face|@media|url\(",
    re.IGNORECASE,
)


def fetch_catalyst_news(max_articles: int = 20) -> list[dict]:
    """Fetch and filter news articles, then prefetch 2-paragraph summaries."""
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
                    # Extract source info for Google News URL resolution
                    source_info = entry.get("source", {})
                    source_domain = source_info.get("href", "")
                    articles.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": feed_name,
                        "source_domain": source_domain,
                        "rss_summary": summary,
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

    # Sort newest first, then by keyword relevance as tiebreaker
    unique.sort(key=lambda x: (_parse_date(x.get("published", "")), len(x["matched_keywords"])), reverse=True)
    top = unique[:max_articles]

    # Prefetch article summaries in parallel
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_resolve_and_summarize, a): a for a in top}
        for future in as_completed(futures):
            article = futures[future]
            try:
                real_url, summary_text = future.result()
                article["real_link"] = real_url or article["link"]
                article["summary_text"] = summary_text
            except Exception:
                article["real_link"] = article["link"]
                article["summary_text"] = ""

    return top


def _resolve_google_news_url(link: str) -> str:
    """Decode a Google News RSS link to the actual publisher URL."""
    if "news.google.com" not in link:
        return link
    try:
        from googlenewsdecoder import new_decoderv1
        result = new_decoderv1(link, interval=1)
        if result.get("status") and result.get("decoded_url"):
            return result["decoded_url"]
    except Exception:
        pass
    return ""


def _resolve_and_summarize(article: dict) -> tuple[str, str]:
    """Resolve real URL and fetch summary for an article."""
    link = article["link"]
    title = article.get("title", "")

    real_url = _resolve_google_news_url(link) or link
    if "news.google.com" in real_url:
        return real_url, ""
    summary = _fetch_and_extract(real_url)
    if summary:
        # Strip leading non-prose junk (bylines, metadata) before first real sentence
        match = re.search(r'(?<![A-Z])[.!?]\s+([A-Z][a-z])', summary)
        if match and match.start() < 150:
            candidate = summary[match.start() + 2:].strip()
            if _has_topic_overlap(candidate, title) and len(candidate) > 100:
                summary = candidate
        # Discard if content is irrelevant to the article
        if title and not _has_topic_overlap(summary, title):
            summary = ""
    return real_url, summary


def _has_topic_overlap(text: str, title: str) -> bool:
    """Check if extracted text shares keywords with the article title."""
    title_words = set(w.lower() for w in re.findall(r'\w{4,}', title))
    text_lower = text.lower()
    matches = sum(1 for w in title_words if w in text_lower)
    return matches >= 2


def _fetch_and_extract(url: str, timeout: int = 10) -> str:
    """Fetch a URL and extract first 2 meaningful paragraphs."""
    if not url:
        return ""
    try:
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return ""
            raw = resp.read(500_000)
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].strip().split(";")[0]
            for encoding in (charset, "utf-8", "latin-1"):
                try:
                    html_text = raw.decode(encoding)
                    break
                except (UnicodeDecodeError, ValueError, LookupError):
                    continue
            else:
                return ""
            return _extract_paragraphs(html_text, count=2)
    except Exception:
        return ""


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _prose_score(text: str) -> float:
    """Score how likely a text block is real article prose (0-1). Generic heuristic."""
    if len(text) < 80:
        return 0.0
    # Boilerplate patterns
    if _BOILERPLATE_RE.search(text):
        return 0.0
    # Must contain a proper sentence ending (lowercase/digit + period)
    if not re.search(r'[a-z0-9][.!]\s', text + " ") and not re.search(r'[a-z0-9][.]$', text):
        return 0.0
    # Reject CSS/JS/code-like content
    if re.search(r'[{};]', text):
        return 0.0
    # Reject non-Latin script blocks (language menus, translation nav)
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total_alpha = len(re.findall(r'\w', text))
    if total_alpha > 0 and latin_chars / total_alpha < 0.7:
        return 0.0
    words = text.split()
    if len(words) < 10:
        return 0.0
    score = 0.0
    # Longer paragraphs score higher (news paragraphs are typically 20-80 words)
    if 15 <= len(words) <= 100:
        score += 0.3
    elif len(words) > 10:
        score += 0.1
    # Sentences ending with periods (prose has these, nav/headlines don't)
    sentences = re.findall(r'[A-Z][^.!?]*[.!?]', text)
    if len(sentences) >= 1:
        score += 0.3
    # Average word length 3-8 (typical English prose)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if 3.5 <= avg_word_len <= 8.0:
        score += 0.2
    # Low ratio of ALL-CAPS words (nav/headers have many)
    caps = sum(1 for w in words if w.isupper() and len(w) > 2)
    if caps / len(words) < 0.15:
        score += 0.2
    return score


def _clean_html(html_text: str) -> str:
    """Remove non-content tags."""
    for tag in ("script", "style", "nav", "header", "footer", "aside",
                "noscript", "svg", "form", "iframe", "figure"):
        html_text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html_text,
                           flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    html_text = re.sub(r"<!--.*?-->", "", html_text, flags=re.DOTALL)
    return html_text


def _extract_paragraphs(html_text: str, count: int = 2) -> str:
    """Extract the best N article paragraphs from HTML using generic heuristics."""
    html_text = _clean_html(html_text)

    # Narrow scope: <article> > <main> > content div > full page
    article_html = html_text
    for tag in ("article", "main"):
        match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html_text,
                          re.DOTALL | re.IGNORECASE)
        if match and len(match.group(1)) > 300:
            article_html = match.group(1)
            break

    # Score all <p> tags and pick the best consecutive run
    paras = re.findall(r"<p[^>]*>(.*?)</p>", article_html,
                       re.DOTALL | re.IGNORECASE)
    scored = []
    for p in paras:
        text = _strip_tags(p).strip()
        sc = _prose_score(text)
        if sc >= 0.5:
            scored.append(text)

    if scored:
        return " ".join(scored[:count])

    # Fallback: extract sentences from the cleaned plain text
    plain = _strip_tags(article_html)
    sentences = re.split(r'(?<=[.!?])\s+', plain)
    result = []
    char_count = 0
    for s in sentences:
        s = s.strip()
        if _prose_score(s) >= 0.4:
            result.append(s)
            char_count += len(s)
            if char_count >= 300:
                break
    if result:
        return " ".join(result[:3])

    return ""
