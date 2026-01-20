# Yahoo Finance News Feed Ingestion

This pipeline ingests weekly Yahoo Finance news headlines for configured tickers and appends new items to a CSV that matches the existing GNews schema.

## What it does
- Fetches the most recent Yahoo Finance news for each configured ticker.
- Normalizes article URLs and computes `id = sha256(normalized_link)`.
- Persists deduped IDs in a SQLite store shared across all tickers.
- Appends new rows to the CSV in a fixed schema.

## Output schema
The CSV header is fixed and must match exactly:
```
id,title,link,published,source,summary,query,fetched_at
```

## Dedupe mechanism
The SQLite database at `state/seen_ids.sqlite` stores IDs in the table:
```
seen_feed_ids(id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)
```
Before appending to the CSV, the script checks this table and skips previously seen IDs.
The `state/seen_ids.sqlite` file is created at runtime and intentionally not committed to git.

## Configuration (add tickers here)
Update `config.yaml` to add new feeds without changing code:
```yaml
feeds:
  - name: bny
    ticker: BK
    query: '("BNY Mellon" OR "Bank of New York Mellon" OR BK) -sports -gossip'
```

## Run locally
From `news_feeds/yahoo_finance/`:
```bash
python ingest.py \
  --config config.yaml \
  --out_csv data/yahoo_finance_news.csv \
  --db_path state/seen_ids.sqlite \
  --max_items 80
```

Single-ticker run without config:
```bash
python ingest.py \
  --ticker BK \
  --query '("BNY Mellon" OR "Bank of New York Mellon" OR BK) -sports -gossip' \
  --out_csv data/yahoo_finance_news.csv \
  --db_path state/seen_ids.sqlite \
  --max_items 80
```

Validate output:
```bash
python validate.py --csv data/yahoo_finance_news.csv
```

## GitHub Actions
The workflow `.github/workflows/yahoo_finance_weekly.yml` runs weekly (UTC) and via manual dispatch. It:
1. Installs requirements from `news_feeds/yahoo_finance/requirements.txt`
2. Runs ingestion and validation
3. Commits and pushes if the CSV or SQLite state changes

## Logging
Logs are structured at INFO level by default. Use `--log_level DEBUG` for verbose output.
