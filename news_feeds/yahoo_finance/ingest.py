#!/usr/bin/env python3
import argparse
import csv
import hashlib
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import yaml
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

BASE_URL = "https://finance.yahoo.com/quote/{ticker}/news/"
FALLBACK_URL = "https://finance.yahoo.com/topic/stock-market-news/"
SCHEMA = [
    "id",
    "title",
    "link",
    "published",
    "source",
    "summary",
    "query",
    "fetched_at",
]


@dataclass
class FeedConfig:
    name: str
    ticker: str
    query: str


@dataclass
class NewsItem:
    title: str
    link: str
    published: str
    source: str
    summary: str


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_config(path: str) -> List[FeedConfig]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    feeds = []
    for entry in data.get("feeds", []):
        if not entry.get("ticker") or not entry.get("query"):
            raise ValueError("Each feed must include ticker and query")
        feeds.append(
            FeedConfig(
                name=entry.get("name", entry["ticker"].lower()),
                ticker=entry["ticker"],
                query=entry["query"],
            )
        )
    return feeds


def should_retry(exception: Exception) -> bool:
    if isinstance(exception, requests.HTTPError) and exception.response is not None:
        return exception.response.status_code in {429} or exception.response.status_code >= 500
    if isinstance(exception, requests.RequestException):
        return exception.response is None
    return False


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(should_retry),
)
def fetch_url(url: str, session: requests.Session) -> Optional[str]:
    logging.debug("Fetching URL: %s", url)
    response = session.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=20,
    )
    if response.status_code == 404:
        logging.warning("Received 404 for %s", url)
        return None
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()
    if 400 <= response.status_code < 500:
        logging.error("Received %s for %s", response.status_code, url)
        return None
    return response.text


def normalize_link(link: str, base_url: str) -> str:
    absolute = urljoin(base_url, link)
    parsed = urlparse(absolute)
    normalized = parsed._replace(query="", fragment="")
    return urlunparse(normalized)


def compute_id(normalized_link: str) -> str:
    return hashlib.sha256(normalized_link.encode("utf-8")).hexdigest()


def extract_time(container: Optional[BeautifulSoup]) -> str:
    if not container:
        return ""
    time_tag = container.find("time")
    if time_tag:
        if time_tag.has_attr("datetime"):
            return time_tag["datetime"].strip()
        return time_tag.get_text(strip=True)
    return ""


def extract_source(container: Optional[BeautifulSoup]) -> str:
    if not container:
        return ""
    source_tag = container.find(["span", "div"], attrs={"class": lambda c: c and "publisher" in c})
    if source_tag:
        return source_tag.get_text(strip=True)
    label = container.find("span", attrs={"data-test": "source"})
    if label:
        return label.get_text(strip=True)
    return ""


def extract_summary(container: Optional[BeautifulSoup]) -> str:
    if not container:
        return ""
    snippet = container.find("p")
    if snippet:
        return snippet.get_text(strip=True)
    return ""


def parse_news_items(html: str, base_url: str) -> List[NewsItem]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    items = []
    seen_links = set()

    for anchor in anchors:
        href = anchor.get("href", "")
        if "/news/" not in href:
            continue
        title = anchor.get_text(strip=True)
        if not title:
            continue
        normalized = normalize_link(href, base_url)
        if normalized in seen_links:
            continue
        container = anchor.find_parent(["article", "li", "div"])
        summary = extract_summary(container)
        published = extract_time(container)
        source = extract_source(container) or "Yahoo Finance"
        items.append(
            NewsItem(
                title=title,
                link=normalized,
                published=published,
                source=source,
                summary=summary,
            )
        )
        seen_links.add(normalized)

    return items


def extract_query_terms(query: str) -> List[str]:
    phrases = []
    start = 0
    while True:
        start_quote = query.find('"', start)
        if start_quote == -1:
            break
        end_quote = query.find('"', start_quote + 1)
        if end_quote == -1:
            break
        phrase = query[start_quote + 1 : end_quote].strip()
        if phrase:
            phrases.append(phrase.lower())
        start = end_quote + 1
    return phrases


def matches_query(item: NewsItem, feed: FeedConfig) -> bool:
    text = f"{item.title} {item.summary}".lower()
    if feed.ticker.lower() in text:
        return True
    for phrase in extract_query_terms(feed.query):
        if phrase in text:
            return True
    return False


def enrich_summary(item: NewsItem, session: requests.Session) -> NewsItem:
    if item.summary:
        return item
    try:
        html = fetch_url(item.link, session)
    except requests.RequestException as exc:
        logging.warning("Failed to fetch article for summary: %s", exc)
        return item
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "description"})
    if not meta:
        meta = soup.find("meta", attrs={"property": "og:description"})
    if meta and meta.get("content"):
        item.summary = meta["content"].strip()
    return item


def ensure_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_feed_ids ("
            "id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)"
        )
        conn.commit()


def load_seen_ids(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT id FROM seen_feed_ids").fetchall()
    return {row[0] for row in rows}


def insert_seen_id(conn: sqlite3.Connection, feed_id: str, first_seen_at: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_feed_ids (id, first_seen_at) VALUES (?, ?)",
        (feed_id, first_seen_at),
    )


def ensure_csv(out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SCHEMA)


def append_rows(out_csv: str, rows: Iterable[List[str]]) -> None:
    with open(out_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def ingest_feed(
    feed: FeedConfig,
    out_csv: str,
    db_path: str,
    max_items: int,
    session: requests.Session,
    fetched_at: str,
) -> int:
    url = BASE_URL.format(ticker=feed.ticker)
    html = fetch_url(url, session)
    source_label = "primary"
    if html is None:
        logging.warning("Primary URL unavailable for %s; using fallback", feed.ticker)
        html = fetch_url(FALLBACK_URL, session)
        source_label = "fallback"
        if html is None:
            logging.error("Fallback URL unavailable for %s; skipping feed", feed.ticker)
            return 0
    items = parse_news_items(html, FALLBACK_URL if source_label == "fallback" else url)
    if source_label == "fallback":
        items = [item for item in items if matches_query(item, feed)]

    if len(items) == 0:
        logging.warning("No items parsed for %s (%s)", feed.ticker, source_label)
    if len(items) > 500:
        logging.warning(
            "Parsed %d items for %s; limiting to first %d",
            len(items),
            feed.ticker,
            max_items,
        )
    items = items[:max_items]
    logging.info("Parsed %d items for %s (%s)", len(items), feed.ticker, source_label)

    ensure_db(db_path)
    ensure_csv(out_csv)

    new_rows = []
    new_count = 0
    skipped = 0

    with sqlite3.connect(db_path) as conn:
        for item in items:
            normalized_link = normalize_link(item.link, url)
            feed_id = compute_id(normalized_link)
            existing = conn.execute(
                "SELECT 1 FROM seen_feed_ids WHERE id = ?", (feed_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue
            item = enrich_summary(item, session)
            insert_seen_id(conn, feed_id, fetched_at)
            new_rows.append(
                [
                    feed_id,
                    item.title,
                    normalized_link,
                    item.published,
                    item.source,
                    item.summary,
                    feed.query,
                    fetched_at,
                ]
            )
            new_count += 1
        conn.commit()

    if new_rows:
        append_rows(out_csv, new_rows)

    logging.info(
        "Feed %s complete: %d new, %d skipped",
        feed.ticker,
        new_count,
        skipped,
    )
    return new_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yahoo Finance news ingestion")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--ticker", help="Single ticker to ingest")
    parser.add_argument("--query", help="Query string for single ticker")
    parser.add_argument(
        "--out_csv",
        default="data/yahoo_finance_news.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--db_path",
        default="state/seen_ids.sqlite",
        help="SQLite DB path",
    )
    parser.add_argument("--max_items", type=int, default=80, help="Max items per ticker")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.ticker and not args.query:
        logging.error("--query is required when using --ticker")
        return 2

    feeds: List[FeedConfig]
    if args.ticker:
        feeds = [FeedConfig(name=args.ticker.lower(), ticker=args.ticker, query=args.query)]
    else:
        feeds = load_config(args.config)
        if not feeds:
            logging.error("No feeds defined in %s", args.config)
            return 2

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()

    total_new = 0
    for feed in feeds:
        try:
            total_new += ingest_feed(
                feed,
                out_csv=args.out_csv,
                db_path=args.db_path,
                max_items=args.max_items,
                session=session,
                fetched_at=fetched_at,
            )
        except requests.RequestException as exc:
            logging.error("Network error for %s: %s", feed.ticker, exc)
            return 1
        except Exception as exc:  # noqa: BLE001
            logging.exception("Unexpected error for %s: %s", feed.ticker, exc)
            return 1

    logging.info("Ingestion complete. Total new items: %d", total_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
