import pandas as pd
from pathlib import Path

# Assumptions:
# 1. Only two files exist: gnews.csv and jpm_press_releases.csv.
# 2. Columns with the same meaning should be normalized to a single canonical name.
# 3. Fields that exist in only one source should still appear in output.
# 4. No column should be dropped implicitly.

DATA_DIR = Path("data")

# Canonical schema and full semantic mapping
COLUMN_MAP = {
    "gnews.csv": {
        "id": "id",
        "title": "title",
        "link": "link",
        "published": "published",
        "source": "source",
        "summary": "summary",
        "query": "query",
        "fetched_at": "fetched_at",
    },
    "jpm_press_releases.csv": {
        "guid": "guid",
        "title": "title",
        "link": "link",
        "published_utc": "published",
        "source": "source",
        "summary": "summary",
        "categories": "categories",
    },
}

# Canonical output columns (union of both schemas)
CANONICAL_COLUMNS = sorted(
    {v for file_map in COLUMN_MAP.values() for v in file_map.values()}
)

def normalize(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if filename not in COLUMN_MAP:
        raise ValueError(f"Unexpected file schema: {filename}")

    # Rename source columns to canonical names
    df = df.rename(columns=COLUMN_MAP[filename])

    # Add missing canonical columns
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Enforce column order
    return df[CANONICAL_COLUMNS]


def collate_csvs() -> pd.DataFrame:
    dfs = []

    for file in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(file)
        df = normalize(df, file.name)
        df["__source_file"] = file.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# Example usage
combined_df = collate_csvs()
combined_df.to_csv("data/collated.csv", index=False)
