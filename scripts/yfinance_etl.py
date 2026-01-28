"""
Yahoo Finance News Scraper
Captures finance-specific news for tickers using yfinance.
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Same entities as your other script
ENTITIES = {
    'BA': 'Boeing',
    'RDDT': 'Reddit',
    'AAPL': 'Apple',
    'NVO': 'Novo Nordisk',
    'DJT': 'Trump Media',
    'TSN': 'Tyson Foods',
    'NVDA': 'Nvidia',
    'NYCB': 'NYCB',
    'CMG': 'Chipotle',
    'TSLA': 'Tesla'
    'BK': 
}

BASE_DIR = 'data'

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def fetch_yahoo_news(entities: dict):
    all_news = []
    run_ts = datetime.now().isoformat()
    
    print(f"Fetching Yahoo Finance news for {len(entities)} tickers...")

    for ticker, name in entities.items():
        try:
            print(f"  > Pulling: {ticker} ({name})")
            
            stock = yf.Ticker(ticker)
            news_items = stock.news
            
            for item in news_items:
                if not item or not isinstance(item, dict):
                        #TODO: log
                        print(f"    [WARNING] Skipped item with no payload for {ticker}, {payload}")
                        continue
                
                payload = item.get('content', item)

                if not payload :
                    #TODO: log
                    print(f"    [WARNING] Skipped item with no payload for {ticker}, {payload}")
                    continue

                published = run_ts # Default fallback
                if 'pubDate' in payload:
                    published = payload['pubDate']
                elif 'providerPublishTime' in payload:
                    published = datetime.fromtimestamp(payload['providerPublishTime']).isoformat()

                link = payload.get('link')
                if not link and 'clickThroughUrl' in payload:
                    link = payload['clickThroughUrl'].get('url')
                if not link and 'canonicalUrl' in payload:
                    link = payload['canonicalUrl'].get('url')

                row = {
                    "id": payload.get('id'),
                    "title": payload.get('title'),
                    "link": link,
                    "published": published,
                    "source": "yfinance",
                    "summary": payload.get('summary'),
                    "query": "",
                    "target_date": datetime.now().strftime('%Y-%m-%d'),
                    "ticker": ticker,
                    "fetched_at": run_ts,                    
                }

                all_news.append(row)
                
            
        except Exception as e:
            print(f"    Error fetching {ticker}: {e}")
            continue

    return pd.DataFrame(all_news)

if __name__ == "__main__":
    target_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(BASE_DIR,target_date,"raw")
    output_file = os.path.join(output_dir, 'yfinance.csv')
    os.makedirs(output_dir, exist_ok=True)

    new_df = fetch_yahoo_news(ENTITIES)
    
    if not new_df.empty:
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            combined_df = pd.concat([existing_df, new_df])
        else:
            combined_df = new_df

        final_df = combined_df.drop_duplicates(subset=['id'])
        final_df.to_csv(output_file, index=False)
        print(f"\nSuccess! Updated {output_file}")
        print(f"Total articles: {len(final_df)}")
    else:
        print("\nNo news found.")