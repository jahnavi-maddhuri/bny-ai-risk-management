"""
Merge Gnews files to remove duplicated articles.
Date range can be configered. 
Fill summary with title if not available.
"""

import pandas as pd
from datetime import datetime, timedelta 

df_list = []
START_DATE = datetime(2025, 4, 1)
for i in range(30):
    DATE_I = START_DATE + timedelta(days=i)
    df_list.append(pd.read_csv(f'data/gnews_summarized/{DATE_I.strftime('%Y-%m-%d')}.csv'))

df = pd.concat(df_list)
df.drop(columns = ['Unnamed: 0'], inplace=True)
df.reset_index(inplace=True, drop=True)
df['summary'] = df['summary'].fillna(df['title'])
df['summary_len'] = df['summary'].str.len()
df['full_text'] = df['full_text'].fillna(df['title'])
df['text_len'] = df['full_text'].str.len()
df = df.loc[df.groupby('id')['text_len'].idxmax()]
df['published_dt'] = pd.to_datetime(df['published'], format='mixed').dt.strftime('%Y-%m-%d')

print("=" * 70)
print("Google News Summarizer")
print("=" * 70)

# save grouped csv
fname = f'data/gnews_multi_day/{START_DATE.strftime('%Y-%m-%d')}--TO--{DATE_I.strftime('%Y-%m-%d')}'
df.to_csv(fname)
print(f'Merged Data saved as: {fname}')