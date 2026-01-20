# GDELT News Feed

This folder contains a self-contained ingestion pipeline for pulling recent company-related
news from the GDELT 2.1 DOC API and emitting a CSV that matches the existing GNews schema.

## What it does

- Queries the GDELT DOC API for each configured feed.
- Normalizes article URLs (removing query parameters and fragments).
- Hashes the normalized URL with SHA-256 for deterministic IDs.
- Appends only new articles to the CSV (deduplicated across runs).

## Configuration

Edit `config.yaml` to add or update feeds. Example:

```yaml
feeds:
  - name: bny_mellon
    query: '"BNY Mellon" OR "Bank of New York Mellon" OR BK'
```

## Usage

```bash
cd news_feeds/gdelt
python ingest.py --config config.yaml
python validate.py --csv data/gdelt_news.csv
```

### Optional flags

- `--max_records 200` controls the max articles per query.
- `--timespan 7d` controls the lookback window (GDELT timespan syntax).

## Output schema

The output CSV schema matches the existing GNews schema exactly:

```
id,title,link,published,source,summary,query,fetched_at
```

## Notes

- The pipeline uses the `data/gdelt_news.csv` file for deduplication across runs.
- The `state/` directory is reserved for future local runtime state and remains empty in Git.
