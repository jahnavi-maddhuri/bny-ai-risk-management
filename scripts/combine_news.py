"""
News Combiner
Merges Yahoo Finance and Google News CSVs into a single daily master file.
"""
import pandas as pd
import os
from datetime import datetime
import glob

# Configuration
BASE_DIR = 'data'
today = datetime.now().strftime('%Y-%m-%d')
MERGED_DIR = os.path.join(BASE_DIR, today,'merged')
RAW_DIR = os.path.join(BASE_DIR, today,'raw')

def normalize_cols(df, source_name):
    """Standardize column names across different scrapers."""

    if df.empty:
        return df
    schema = {
        'id': 'id',
        'title': 'title',
        'link': 'link',
        'published': 'published',
        'source':'source',
        'summary': 'summary',
        'target_date':'target_date',
        'ticker': 'ticker',
        'query':'query',
        'fetched_at': 'fetched_at'
    }
        
    # Ensure all columns exist
    for col in schema.values():
        if col not in df.columns:
            df[col] = None
            
    return df

def run_merge():
    print(f"Running merge for {today}...")
    
    all_dfs = []

    # 1. Load Yahoo Data (Today)
    y_path = os.path.join(RAW_DIR, 'yfinance.csv')
    print(y_path)
    if os.path.exists(y_path):
        try:
            ydf = pd.read_csv(y_path)
            ydf = normalize_cols(ydf, 'yahoo')
            all_dfs.append(ydf)
            print(f"  Loaded {len(ydf)} Yahoo articles.")
        except Exception as e:
            print(f"  Error loading Yahoo: {e}")

    # 2. Load GNews Data (Today)
    # GNews script usually saves as 'data/gnews/YYYY-MM-DD.csv'
    g_path = os.path.join(RAW_DIR, f"gnews.csv")
    if os.path.exists(g_path):
        try:
            gdf = pd.read_csv(g_path)
            gdf = normalize_cols(gdf, 'gnews')
            all_dfs.append(gdf)
            print(f"  Loaded {len(gdf)} GNews articles.")
        except Exception as e:
            print(f"  Error loading GNews: {e}")

    # 3. Merge & Deduplicate
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Deduplicate by Link (most robust) or Title
        before = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['link'])
        after = len(merged_df)
        #TODO log
        print(f"  Merged total: {before} -> {after} (removed {before-after} duplicates)")

        # Save
        os.makedirs(MERGED_DIR, exist_ok=True)
        out_file = os.path.join(MERGED_DIR, f"merged.csv")
        merged_df.to_csv(out_file, index=False)
        print(f"Success! Saved merged data to {out_file}")
    else:
        print("No data found for today from either source.")

if __name__ == "__main__":
    run_merge()