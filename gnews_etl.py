"""
Google News Entity Scraper
Captures news articles for specified entities over configurable date ranges.
"""

import os
import hashlib
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from dateutil import tz
import pandas as pd
from pygooglenews import GoogleNews

# ============================================================================
# CONFIGURATION SECTION - Modify these to change behavior
# ============================================================================

# Entities to track (ticker symbols and company names)
ENTITIES = {
    'BA': ['Boeing', 'BA'],
    'RDDT': ['Reddit', 'RDDT'],
    'AAPL': ['Apple', 'AAPL'],
    'NVO': ['Novo Nordisk', 'NVO'],
    'DJT': ['Trump Media', 'DJT', 'Truth Social'],
    'TSN': ['Tyson Foods', 'TSN'],
    'NVDA': ['Nvidia', 'NVDA'],
    'NYCB': ['NYCB', 'New York Community Bank'],
    'CMG': ['Chipotle', 'CMG'],
    'TSLA': ['Tesla', 'TSLA']
}

# Optional: Add News topics/categories to search for each entity
NEWS_TOPICS = [
    'earnings',
    'financial results',
    'acquisition',
    'merger',
    'product launch',
    'regulatory',
    'lawsuit',
    'stock price',
    'leadership change',
    'strategy',
    # Add more topics as needed
]

# API configuration
LANG = "en"
COUNTRY = "US"

# Directory configuration
BASE_DIR = 'data'
STATE_DIR = os.path.join(BASE_DIR, 'state')
SEEN_PATH = os.path.join(STATE_DIR, 'gnews_seen_ids.csv')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_entity_queries(entities: dict, topics: list = None) -> list:
    """
    Build search queries for entities.
    
    Args:
        entities: Dict mapping ticker to list of name variants
        topics: Optional list of topics to combine with entity names
    
    Returns:
        List of search query strings
    """
    queries = []
    
    for ticker, names in entities.items():
        # Base entity query (any news about the entity)
        name_parts = ' OR '.join(f'"{name}"' for name in names)
        base_query = f'({name_parts})'
        
        # Exclude noise
        exclusions = ' -gossip -celebrity -"fantasy sports" -"game recap"'
        
        if topics:
            # Create topic-specific queries
            for topic in topics:
                query = f'{base_query} AND "{topic}"{exclusions}'
                queries.append(query)
        else:
            # Just search for the entity
            queries.append(base_query + exclusions)
    
    return queries


def clean_url(url: str) -> str:
    """Remove common tracking params to get canonical URL."""
    try:
        p = urlparse(url)
        if not p.scheme:
            return url
        # Filter out tracking parameters
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not re.match(r'^(utm_|gclid|fbclid|mc_|ref)', k, re.I)]
        return urlunparse((p.scheme, p.netloc, p.path, "", urlencode(q), ""))
    except Exception:
        return url


def make_id(title: str, canonical_link: str) -> str:
    """Generate unique ID for article based on title and link."""
    base = f"{(title or '').strip()}|{(canonical_link or '').strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def load_seen(path: str) -> set:
    """Load set of previously seen article IDs."""
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path)["id"].astype(str).tolist())
    except Exception as e:
        print(f"Warning: Could not load seen IDs from {path}: {e}")
        return set()


def append_seen(path: str, new_ids: list, run_ts: str) -> None:
    """Append newly seen article IDs to tracking file."""
    if not new_ids:
        return
    df = pd.DataFrame({"id": new_ids, "first_seen_at": run_ts})
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def write_master(rows: list, master_path: str) -> None:
    """Append new articles to master CSV."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(master_path)
    df.to_csv(master_path, mode="a", header=header, index=False)


def extract_ticker(query):
    match = re.search(r'OR\s+"([^"]+)"', query)
    return match.group(1) if match else None

# ============================================================================
# MAIN SCRAPING FUNCTIONS
# ============================================================================

def run_pull_for_date(target_date: datetime, queries: list, entities: dict) -> pd.DataFrame:
    """
    Pull news for a specific date.
    
    Args:
        target_date: The date to pull news for
        queries: List of search queries
        entities: Entity configuration dict
    
    Returns:
        DataFrame of new articles
    """
    gn = GoogleNews(lang=LANG)
    seen = load_seen(SEEN_PATH)
    run_iso = datetime.now(tz=tz.tzlocal()).isoformat()
    
    # Set up date range for this specific day
    from_date = target_date.strftime('%Y-%m-%d')
    to_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    raw_rows, master_rows, new_ids = [], [], []
    print(f"\nFetching news for {from_date}...")
    
    for i, q in enumerate(queries, 1):
            print(f"  Query {i}/{len(queries)}: {q[:80]}...")
            
            try:
                res = gn.search(q, from_=from_date, to_=to_date)
                entries = res.get("entries")
                raw_rows = []
                for e in entries:
                    title = e.get("title", "")
                    link = e.get("link", "")
                    src = (e.get("source", {}) or {}).get("title", "")
                    published = e.get("published", "")
                    canonical = clean_url(link)
                    _id = make_id(title, canonical)
                    # entity = extract_entity_from_query(q, entities)
                    
                    row = {
                        "id": _id,
                        "title": title,
                        "link": canonical,
                        "published": published,
                        "source": src,
                        "summary": '',
                        "query": q,
                        "target_date": from_date,
                        "ticker": extract_ticker(q),
                        "fetched_at": run_iso
                    }
                    raw_rows.append(row)
                
                    if _id not in seen:
                        master_rows.append(row)
                        new_ids.append(_id)
                print(f'\t {len(master_rows)} new articles through query {i}.')
            except Exception as e:
                print(f"    Error with query: {e}")
                continue
    
    # Update seen IDs
    append_seen(SEEN_PATH, new_ids, run_iso)
    
    # Return new articles
    print(f"  Found {len(master_rows)} new articles total.")
    gnews_path = os.path.join(BASE_DIR, 
                              f'gnews/{target_date.strftime('%Y-%m-%d')}.csv')

    write_master(master_rows, gnews_path)
    return pd.DataFrame(master_rows)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Google News Entity Scraper")
    print("=" * 70)
    # # Date range configuration
    START_DATE = datetime(2024, 3, 11)

    queries = build_entity_queries(ENTITIES, topics=None)
    df = run_pull_for_date(START_DATE, queries, ENTITIES)