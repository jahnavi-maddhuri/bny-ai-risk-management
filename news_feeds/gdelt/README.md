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
  max_hf_new_per_run: 10
  max_hf_backfill_per_run: 20
  backfill_missing_summaries: true
  backfill_limit: 30
  denylist_domains:
    - dailypolitical.com
    - themarketsdaily.com
    - tickerreport.com
  hf_timeout_s: 18
  hf_delay_s: 0.8
  hf_model_primary: sshleifer/distilbart-cnn-12-6
  hf_model_fallback: facebook/bart-large-cnn
```

### Summary fallback chain

The ingestion pipeline fills the `summary` column using the following best-effort steps:

1. Use GDELT `snippet` or `description` if available.
2. Fetch the article URL and extract main content for summarization:
   - Prefer `<article>` text, then `main`/`[role="main"]`, then all paragraphs.
   - Drop boilerplate paragraphs (e.g., “Read … at …”, subscribe/sign-up prompts).
   - Ignore very short paragraphs (<40 chars) to skip nav/footer/share noise.
   - Prefer the first 3–6 meaningful paragraphs to reduce recommendation noise.
   - Cap extracted text to ~4,000 characters.
3. If extracted text is short but a long `meta`/`og:description` exists, use that for HF
   summarization as a fallback input.
4. If summarization fails or is low quality, fall back to `meta`/`og:description` when
   available. The HF call is best-effort and may be rate-limited.
   - Model order: `sshleifer/distilbart-cnn-12-6`, then `facebook/bart-large-cnn`.

The job never fails if summarization does; it will continue with empty summaries as needed.

You can tune the summarization behavior in `config.yaml`:
- `max_hf_new_per_run`: limit HF summary attempts for newly ingested rows.
- `max_hf_backfill_per_run`: limit HF summary attempts for backfill rows.
- `backfill_missing_summaries`: enable best-effort backfill for older rows with empty summaries.
- `backfill_limit`: cap how many existing rows are updated per run (older rows will
  continue to be processed across weekly runs).
- `denylist_domains`: skip fetching/summarizing known low-quality domains during backfill.
- `hf_timeout_s`: request timeout for HF calls and article fetches.
- `hf_delay_s`: delay between HF calls to reduce rate limiting (default: 0.8s).
- Summarization parameters: `max_length=120`, `min_length=40`, `do_sample=false`.

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
- `--max_summaries_per_run 10` limits how many new rows attempt HF summarization per run.
- `--[no-]backfill_missing_summaries` toggles the backfill step for existing rows.
- `--backfill_limit 30` limits how many existing rows are updated per run.

## Output schema

The output CSV schema matches the existing GNews schema exactly:

```
id,title,link,published,source,summary,query,fetched_at
```

## Notes

- The pipeline uses the `data/gdelt_news.csv` file for deduplication across runs.
- The `state/` directory is reserved for future local runtime state and remains empty in Git.
