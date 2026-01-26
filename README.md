# BNY-AI-Risk-Management
This repository is for our Duke MIDS (Masters in Interdisciplinary Data Science) Capstone project with BNY Mellon. As we are focused on developing an AI-driven solution for risk management, the repo includes analysis, modeling, and product code.

## GDELT weekly backfill behavior

The GDELT ingestion workflow (`news_feeds/gdelt`) appends new articles and then backfills
older rows with empty summaries on each weekly run. Backfill is intentionally incremental:
the pipeline selects up to `summary.backfill_limit` oldest missing summaries each run,
so historical summaries fill in gradually over multiple weeks. Tune the limits in
`news_feeds/gdelt/config.yaml`:

- `summary.max_hf_new_per_run`: Hugging Face summary attempts reserved for newly ingested rows.
- `summary.max_hf_backfill_per_run`: Hugging Face summary attempts reserved for backfill rows.
- `summary.backfill_limit`: how many historical rows to attempt per run (raise to accelerate
  backfill, lower to reduce runtime).
- `summary.denylist_domains`: domains skipped during backfill when they are low quality.

## GDELT ingestion resiliency

The GDELT DOC API can be flaky. The ingestion job is designed to retry transient network
errors and skip a feed if retries still fail, so the workflow keeps running and will
recover on the next scheduled run. Use the `--fail_on_feed_error` flag if you need the
job to exit non-zero when a feed fetch fails after retries (default: false). Keep this
flag disabled in CI to avoid workflow failures from temporary API issues.
