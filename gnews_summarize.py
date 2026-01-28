"""
Takes the links found from queries and produce summaries.
(1) takes the url in 'link' and produces the real url in 'actual_link'
(2) finds the text in the 'actual_link' and put in 'full_text'
(3) summarizes using hugging face and outputs in 'summary' 
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import time
import pandas as pd

def get_real_url_selenium(google_url):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(google_url)
        time.sleep(2)  # Wait for redirect
        actual_url = driver.current_url
        driver.quit()
        return actual_url
    except:
        driver.quit()
        return google_url

def extract_text(url):
    if url == None:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text if text else None
    except:
        return None

summarizer = pipeline(
    task="summarization",
    model=AutoModelForSeq2SeqLM.from_pretrained("KedarPanchal/flan-t5-small-summary-finetune"),
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-small")
)

def summarize_text(text):
    text = str(text)
    if text == None or text == 'nan' or len(text) == 0:
        return None
    
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.3:  # More than 30% non-ASCII
        return None
    
    try:
        s = summarizer(text, min_new_tokens=10, do_sample=False)
        return s[0]['summary_text']
    except:
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("Google News Summarizer")
    print("=" * 70)

    # # Date range configuration
    START_DATE = datetime(2025, 4, 2) # to 4/10
    df = pd.read_csv(f'data/gnews/{START_DATE.strftime('%Y-%m-%d')}.csv')
    df['actual_link'] = df['link'].apply(get_real_url_selenium)
    df['full_text'] = df['actual_link'].apply(extract_text)
    df['summary'] = df['full_text'].apply(summarize_text)
    df.to_csv(f'data/gnews_summarized/{START_DATE.strftime('%Y-%m-%d')}.csv')
    print(f'{df.shape[0]} Articles total in file.')
    print(f'{df[df['summary'].notnull()].shape[0]} summaries produced.')