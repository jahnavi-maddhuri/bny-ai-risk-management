#!/usr/bin/env python3
import argparse
import csv
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional
from urllib.parse import urlparse, urlunparse

import requests
import yaml
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
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
        if not entry.get("query"):
            raise ValueError("Each feed must include a query")
        feeds.append(
            FeedConfig(
                name=entry.get("name", entry["query"].lower()),
                query=entry["query"],
            )
        )
    return feeds


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(requests.RequestException),
)
def fetch_feed(feed: FeedConfig, max_records: int, timespan: str) -> List[dict]:
    params = {
        "query": feed.query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "HybridRel",
        "timespan": timespan,
    }
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "bny-ai-risk-management/1.0",
            "Accept": "application/json",
        },
    )
    response.raise_for_status()
    data = response.json()
    return data.get("articles", [])


def normalize_link(link: str) -> str:
    parsed = urlparse(link.strip())
    normalized = parsed._replace(query="", fragment="")
    return urlunparse(normalized)


def compute_id(normalized_link: str) -> str:
    return hashlib.sha256(normalized_link.encode("utf-8")).hexdigest()


def ensure_csv(out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SCHEMA)


def load_existing_ids(out_csv: str) -> set:
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header != SCHEMA:
            raise ValueError("Existing CSV schema mismatch; run validate.py to inspect.")
        return {row[0] for row in reader if row}


def append_rows(out_csv: str, rows: Iterable[List[str]]) -> None:
    with open(out_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def build_item(article: dict) -> NewsItem:
    title = article.get("title", "").strip()
    link = article.get("url", "").strip()
    published = article.get("seendate") or article.get("datetime") or ""
    source = (
        article.get("source")
        or article.get("domain")
        or article.get("sourcecountry")
        or "GDELT"
    )
    summary = article.get("snippet", "").strip()
    return NewsItem(
        title=title,
        link=link,
        published=published,
        source=source,
        summary=summary,
    )


def ingest_feed(
    feed: FeedConfig,
    out_csv: str,
    max_records: int,
    timespan: str,
    fetched_at: str,
    seen_ids: set,
) -> int:
    try:
        articles = fetch_feed(feed, max_records=max_records, timespan=timespan)
    except requests.RequestException as exc:
        logging.error("Failed to fetch feed %s: %s", feed.name, exc)
        return 0
    except ValueError as exc:
        logging.error("Invalid response for %s: %s", feed.name, exc)
        return 0

    if not articles:
        logging.warning("No articles found for %s", feed.name)
        return 0

    new_rows: List[List[str]] = []
    new_count = 0
    skipped = 0

    for article in articles:
        item = build_item(article)
        if not item.link or not item.title:
            continue
        normalized_link = normalize_link(item.link)
        feed_id = compute_id(normalized_link)
        if feed_id in seen_ids:
            skipped += 1
            continue
        seen_ids.add(feed_id)
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

    if new_rows:
        append_rows(out_csv, new_rows)

    logging.info("Feed %s complete: %d new, %d skipped", feed.name, new_count, skipped)
    return new_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GDELT DOC API news ingestion")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--name", help="Single feed name")
    parser.add_argument("--query", help="Query string for single feed")
    parser.add_argument(
        "--out_csv",
        default="data/gdelt_news.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=200,
        help="Max records per feed",
    )
    parser.add_argument(
        "--timespan",
        default="7d",
        help="Time window for GDELT (e.g. 7d, 10d)",
    )
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

    feeds: List[FeedConfig]
    if args.query:
        feeds = [FeedConfig(name=args.name or "single_feed", query=args.query)]
    else:
        feeds = load_config(args.config)
        if not feeds:
            logging.error("No feeds defined in %s", args.config)
            return 2

    ensure_csv(args.out_csv)
    try:
        seen_ids = load_existing_ids(args.out_csv)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    fetched_at = datetime.now(timezone.utc).isoformat()
    total_new = 0

    for feed in feeds:
        total_new += ingest_feed(
            feed,
            out_csv=args.out_csv,
            max_records=args.max_records,
            timespan=args.timespan,
            fetched_at=fetched_at,
            seen_ids=seen_ids,
        )

    logging.info("Ingestion complete. Total new items: %d", total_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
