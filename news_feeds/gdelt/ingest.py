#!/usr/bin/env python3
import argparse
import csv
import hashlib
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import yaml
from bs4 import BeautifulSoup
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
SUMMARY_TEXT_LIMIT = 4000
HF_SUMMARY_MAX_LENGTH = 120
HF_SUMMARY_MIN_LENGTH = 40
HF_DEFAULT_TIMEOUT_S = 18
HF_DEFAULT_DELAY_S = 0.8
HF_PRIMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
HF_FALLBACK_MODEL = "facebook/bart-large-cnn"
MIN_ARTICLE_CHARS = 500
MIN_PARAGRAPH_CHARS = 40
MIN_SUMMARY_WORDS = 20


@dataclass
class FeedConfig:
    name: str
    query: str


@dataclass
class SummaryConfig:
    enable_hf_summary: bool = True
    max_summaries_per_run: int = 10
    backfill_missing_summaries: bool = True
    backfill_limit: int = 10
    hf_timeout_s: int = HF_DEFAULT_TIMEOUT_S
    hf_delay_s: float = HF_DEFAULT_DELAY_S
    hf_model_primary: str = HF_PRIMARY_MODEL
    hf_model_fallback: str = HF_FALLBACK_MODEL


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


def load_summary_config(path: str) -> SummaryConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    return SummaryConfig(
        enable_hf_summary=summary.get("enable_hf_summary", True),
        max_summaries_per_run=int(summary.get("max_summaries_per_run", 10)),
        backfill_missing_summaries=summary.get("backfill_missing_summaries", True),
        backfill_limit=int(summary.get("backfill_limit", 10)),
        hf_timeout_s=int(summary.get("hf_timeout_s", HF_DEFAULT_TIMEOUT_S)),
        hf_delay_s=float(summary.get("hf_delay_s", HF_DEFAULT_DELAY_S)),
        hf_model_primary=summary.get("hf_model_primary", HF_PRIMARY_MODEL),
        hf_model_fallback=summary.get("hf_model_fallback", HF_FALLBACK_MODEL),
    )


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
    summary = (article.get("snippet") or article.get("description") or "").strip()
    return NewsItem(
        title=title,
        link=link,
        published=published,
        source=source,
        summary=summary,
    )


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    for fmt in (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d%H%M%S",
    ):
        try:
            if fmt:
                parsed = datetime.strptime(cleaned, fmt)
            else:
                parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue
    return None


def is_boilerplate_paragraph(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith("read "):
        return True
    return bool(re.search(r"\bread\b.*\bat\b", lowered))


def collect_paragraphs(container: BeautifulSoup) -> List[str]:
    paragraphs: List[str] = []
    for paragraph in container.find_all("p"):
        text = normalize_text(paragraph.get_text(" ", strip=True))
        if len(text) < MIN_PARAGRAPH_CHARS:
            continue
        if is_boilerplate_paragraph(text):
            continue
        paragraphs.append(text)
    return paragraphs


def extract_candidate_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    meta_summary = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_summary = normalize_text(meta["content"])[:SUMMARY_TEXT_LIMIT]
    og_meta = soup.find("meta", attrs={"property": "og:description"})
    if og_meta and og_meta.get("content"):
        meta_summary = normalize_text(og_meta["content"])[:SUMMARY_TEXT_LIMIT]

    candidate_paragraphs: List[str] = []
    for selector in ("article", "main", '[role="main"]'):
        container = soup.select_one(selector)
        if container:
            candidate_paragraphs = collect_paragraphs(container)
            if candidate_paragraphs:
                break
    if not candidate_paragraphs:
        candidate_paragraphs = collect_paragraphs(soup)
    candidate = normalize_text(" ".join(candidate_paragraphs))[:SUMMARY_TEXT_LIMIT]
    return candidate, meta_summary


def fetch_article_text(url: str, timeout: int) -> Tuple[str, str]:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "bny-ai-risk-management/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.debug("Failed to fetch article %s: %s", url, exc)
        return "", ""
    return extract_candidate_text(response.text)


def is_summary_usable(summary: Optional[str]) -> bool:
    if not summary:
        return False
    words = summary.split()
    if len(words) < MIN_SUMMARY_WORDS:
        return False
    lowered = summary.lower()
    if "read " in lowered or "click" in lowered:
        return False
    return True


def hf_summarize(text: str, config: SummaryConfig) -> Optional[str]:
    if not text:
        return None
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": HF_SUMMARY_MAX_LENGTH,
            "min_length": HF_SUMMARY_MIN_LENGTH,
            "do_sample": False,
        },
    }
    for model_id in [config.hf_model_primary, config.hf_model_fallback]:
        for attempt in range(2):
            try:
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model_id}",
                    json=payload,
                    timeout=config.hf_timeout_s,
                )
            except requests.RequestException as exc:
                logging.debug("HF request error (%s): %s", model_id, exc)
                if attempt == 1:
                    break
                continue
            if response.status_code in {429, 503}:
                logging.debug("HF rate limited (%s): %s", model_id, response.status_code)
                break
            if response.status_code >= 400:
                logging.debug("HF error (%s): %s", model_id, response.status_code)
                if attempt == 1:
                    break
                continue
            try:
                data = response.json()
            except ValueError:
                logging.debug("HF invalid JSON response (%s)", model_id)
                if attempt == 1:
                    break
                continue
            if isinstance(data, dict) and data.get("error"):
                logging.debug("HF model error (%s): %s", model_id, data.get("error"))
                break
            if isinstance(data, list) and data:
                summary_text = data[0].get("summary_text") if isinstance(data[0], dict) else None
                if summary_text:
                    return normalize_text(summary_text)
            break
    return None


def backfill_missing_summaries(out_csv: str, summary_config: SummaryConfig) -> None:
    if not summary_config.backfill_missing_summaries:
        logging.info("Backfill disabled; skipping missing summary backfill.")
        return
    if not os.path.exists(out_csv):
        logging.warning("Backfill skipped; CSV not found at %s", out_csv)
        return
    with open(out_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header != SCHEMA:
            logging.error("Backfill aborted; CSV header mismatch in %s", out_csv)
            return
        rows = list(reader)

    summary_index = SCHEMA.index("summary")
    link_index = SCHEMA.index("link")
    published_index = SCHEMA.index("published")
    fetched_index = SCHEMA.index("fetched_at")

    empty_indices: List[Tuple[int, datetime]] = []
    for idx, row in enumerate(rows):
        summary_value = row[summary_index] if len(row) > summary_index else ""
        if not summary_value.strip():
            published = row[published_index] if len(row) > published_index else ""
            fetched = row[fetched_index] if len(row) > fetched_index else ""
            parsed = parse_datetime(published) or parse_datetime(fetched) or datetime.min.replace(
                tzinfo=timezone.utc
            )
            empty_indices.append((idx, parsed))

    empty_before = len(empty_indices)
    logging.info("Backfill candidates with empty summary: %d", empty_before)
    if empty_before == 0:
        return

    empty_indices.sort(key=lambda item: item[1], reverse=True)
    selected_indices = [idx for idx, _ in empty_indices[: summary_config.backfill_limit]]

    extracted_content_ok = 0
    skipped_no_content = 0
    hf_success = 0
    hf_failed = 0
    fell_back_to_meta = 0
    still_empty = 0
    updated = 0
    hf_budget = summary_config.max_summaries_per_run

    for idx in selected_indices:
        row = rows[idx]
        link = row[link_index] if len(row) > link_index else ""
        if not link:
            still_empty += 1
            continue
        candidate, meta_summary = fetch_article_text(link, summary_config.hf_timeout_s)
        summary = ""
        if candidate and len(candidate) >= MIN_ARTICLE_CHARS:
            extracted_content_ok += 1
            hf_summary = None
            attempted_hf = False
            if summary_config.enable_hf_summary and hf_budget > 0:
                attempted_hf = True
                hf_summary = hf_summarize(candidate, summary_config)
                hf_budget -= 1
                time.sleep(summary_config.hf_delay_s)
            if is_summary_usable(hf_summary):
                summary = hf_summary
                hf_success += 1
            elif attempted_hf:
                hf_failed += 1
        else:
            skipped_no_content += 1
        if not summary and meta_summary:
            summary = meta_summary
            fell_back_to_meta += 1
        elif summary_config.enable_hf_summary:
            logging.debug(
                "Backfill skipped HF summary for %s; insufficient extracted text.",
                link,
            )

        if summary:
            row[summary_index] = summary
            updated += 1
        else:
            still_empty += 1

    if updated:
        directory = os.path.dirname(out_csv) or "."
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=directory,
            delete=False,
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(SCHEMA)
            writer.writerows(rows)
            temp_name = handle.name
        os.replace(temp_name, out_csv)

    logging.info(
        "Backfill complete: filled=%d extracted_content_ok=%d skipped_no_content=%d "
        "hf_success=%d hf_failed=%d fell_back_to_meta=%d still_empty=%d",
        updated,
        extracted_content_ok,
        skipped_no_content,
        hf_success,
        hf_failed,
        fell_back_to_meta,
        still_empty,
    )


def ingest_feed(
    feed: FeedConfig,
    out_csv: str,
    max_records: int,
    timespan: str,
    fetched_at: str,
    seen_ids: set,
    summary_config: SummaryConfig,
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
    summary_budget = summary_config.max_summaries_per_run
    hf_budget = summary_config.max_summaries_per_run
    summary_counts = {
        "from_gdelt_snippet": 0,
        "extracted_content_ok": 0,
        "skipped_no_content": 0,
        "hf_success": 0,
        "hf_failed": 0,
        "fell_back_to_meta": 0,
        "still_empty": 0,
    }

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
        summary_source = "from_gdelt_snippet" if item.summary else ""
        if not item.summary and summary_budget > 0:
            candidate, meta_summary = fetch_article_text(
                normalized_link, summary_config.hf_timeout_s
            )
            if candidate and len(candidate) >= MIN_ARTICLE_CHARS:
                summary_counts["extracted_content_ok"] += 1
                attempted_hf = False
                hf_summary = None
                if summary_config.enable_hf_summary and hf_budget > 0:
                    attempted_hf = True
                    hf_summary = hf_summarize(candidate, summary_config)
                    hf_budget -= 1
                    time.sleep(summary_config.hf_delay_s)
                    if is_summary_usable(hf_summary):
                        item.summary = hf_summary
                        summary_counts["hf_success"] += 1
                    elif attempted_hf:
                        summary_counts["hf_failed"] += 1
            else:
                summary_counts["skipped_no_content"] += 1
            if not item.summary and meta_summary:
                item.summary = meta_summary
                summary_counts["fell_back_to_meta"] += 1
            summary_budget -= 1

        if not item.summary:
            summary_counts["still_empty"] += 1
        elif summary_source:
            summary_counts[summary_source] += 1

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
    logging.info(
        "Summary stats for %s: gdelt=%d extracted_content_ok=%d skipped_no_content=%d "
        "hf_success=%d hf_failed=%d fell_back_to_meta=%d still_empty=%d",
        feed.name,
        summary_counts["from_gdelt_snippet"],
        summary_counts["extracted_content_ok"],
        summary_counts["skipped_no_content"],
        summary_counts["hf_success"],
        summary_counts["hf_failed"],
        summary_counts["fell_back_to_meta"],
        summary_counts["still_empty"],
    )
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
    parser.add_argument(
        "--enable_hf_summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Hugging Face summarization for missing summaries",
    )
    parser.add_argument(
        "--max_summaries_per_run",
        type=int,
        default=10,
        help="Max number of items per run to attempt summarization for",
    )
    parser.add_argument(
        "--backfill_missing_summaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backfill empty summaries in existing CSV rows",
    )
    parser.add_argument(
        "--backfill_limit",
        type=int,
        default=10,
        help="Max number of existing rows to backfill per run",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    feeds: List[FeedConfig]
    if args.query:
        feeds = [FeedConfig(name=args.name or "single_feed", query=args.query)]
        summary_config = SummaryConfig(
            enable_hf_summary=args.enable_hf_summary,
            max_summaries_per_run=args.max_summaries_per_run,
            backfill_missing_summaries=args.backfill_missing_summaries,
            backfill_limit=args.backfill_limit,
        )
    else:
        feeds = load_config(args.config)
        if not feeds:
            logging.error("No feeds defined in %s", args.config)
            return 2
        summary_config = load_summary_config(args.config)
        summary_config.enable_hf_summary = args.enable_hf_summary
        summary_config.max_summaries_per_run = args.max_summaries_per_run
        summary_config.backfill_missing_summaries = args.backfill_missing_summaries
        summary_config.backfill_limit = args.backfill_limit

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
            summary_config=summary_config,
        )

    backfill_missing_summaries(args.out_csv, summary_config)

    logging.info("Ingestion complete. Total new items: %d", total_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
