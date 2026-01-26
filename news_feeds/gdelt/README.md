# GDELT News Feed

This folder contains a self-contained ingestion pipeline for pulling recent company-related
news from the GDELT 2.1 DOC API and emitting a CSV that matches the existing GNews schema.

## What it does

- Queries the GDELT DOC API for each configured feed.
- Normalizes article URLs (removing query parameters and fragments).
- Hashes the normalized URL with SHA-256 for deterministic IDs.
- Appends only new articles to the CSV (deduplicated across runs).
- Populates the `summary` column using a fallback chain that does not require any secrets.

## Configuration

Edit `config.yaml` to add or update feeds. Example:

```yaml
feeds:
  - name: bny_mellon
    query: '"BNY Mellon" OR "Bank of New York Mellon" OR BK'
summary:
  enable_hf_summary: true
  max_summaries_per_run: 10
  hf_timeout_s: 18
  hf_delay_s: 0.7
  hf_model_primary: facebook/bart-large-cnn
  hf_model_fallback: sshleifer/distilbart-cnn-12-6
```

### Summary fallback chain

The ingestion pipeline fills the `summary` column using the following best-effort steps:

1. Use GDELT `snippet` or `description` if available.
2. Fetch the article URL and extract a candidate summary:
   - Prefer `meta[name="description"]` or `og:description`.
   - Otherwise, concatenate the first 2–4 `<p>` paragraphs.
   - Cap extracted text to ~2,000 characters.
3. If still empty, call the Hugging Face hosted inference endpoint (no auth) to summarize
   the extracted text. This is best-effort and may be rate-limited.

The job never fails if summarization does; it will continue with empty summaries as needed.

You can tune the summarization behavior in `config.yaml`:

- `max_summaries_per_run`: limit how many missing summaries are processed per run.
- `hf_timeout_s`: request timeout for HF calls and article fetches.
- `hf_delay_s`: delay between HF calls to reduce rate limiting.

## Usage

```bash
cd news_feeds/gdelt
python ingest.py --config config.yaml
python validate.py --csv data/gdelt_news.csv
```

### Optional flags

- `--max_records 200` controls the max articles per query.
- `--timespan 7d` controls the lookback window (GDELT timespan syntax).
- `--[no-]enable_hf_summary` toggles Hugging Face summarization (default: enabled).
- `--max_summaries_per_run 10` limits how many missing summaries are processed per run to
  reduce rate limiting.

## Output schema

The output CSV schema matches the existing GNews schema exactly:

```
id,title,link,published,source,summary,query,fetched_at
```

## Notes

- The pipeline uses the `data/gdelt_news.csv` file for deduplication across runs.
- The `state/` directory is reserved for future local runtime state and remains empty in Git.
