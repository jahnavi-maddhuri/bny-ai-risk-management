"""
Optimized version for Google Colab
- Reuses single browser session
- Batches summarization on GPU
- Parallel processing where possible
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from datetime import datetime
import time
import pandas as pd
import torch
from tqdm import tqdm

# ============= OPTIMIZED URL FETCHING =============
class URLFetcher:
    """Reuses single browser instance for all URLs"""
    def __init__(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = webdriver.Chrome(options=options)
    
    def get_real_url(self, google_url):
        try:
            self.driver.get(google_url)
            WebDriverWait(self.driver, 10).until(
                lambda d: 'news.google.com' not in d.current_url
            )
            return self.driver.current_url
        except Exception:
            return google_url
    
    def close(self):
        self.driver.quit()

def get_all_urls(df):
    """Process all URLs with single browser instance"""
    fetcher = URLFetcher()
    actual_links = []
    
    for url in tqdm(df['url'], desc="Fetching URLs"):
        actual_links.append(fetcher.get_real_url(url))
        time.sleep(0.5)  # Small delay to avoid being blocked
    
    fetcher.close()
    return actual_links

# ============= OPTIMIZED TEXT EXTRACTION =============
def extract_text(url):
    if pd.isna(url) or url == ''or"news.google.com"in url:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text if text else None
    except Exception:
        return None

# ============= OPTIMIZED SUMMARIZATION (GPU BATCHED) =============
class BatchSummarizer:
    """Batched summarization on GPU for speed"""
    def __init__(self, batch_size=8):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "KedarPanchal/flan-t5-small-summary-finetune"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.batch_size = batch_size
    
    def is_valid_text(self, text):
        if pd.isna(text) or text == '' or len(str(text)) == 0:
            return False
        text = str(text)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.3:
            return False
        return True
    
    def summarize_batch(self, texts):
        """Summarize multiple texts at once"""
        valid_indices = [i for i, t in enumerate(texts) if self.is_valid_text(t)]
        valid_texts = [str(texts[i]) for i in valid_indices]
        
        if not valid_texts:
            return [None] * len(texts)
        
        inputs = self.tokenizer(
            valid_texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=10,
                max_new_tokens=100,
                do_sample=False
            )
        
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        result = [None] * len(texts)
        for idx, summary in zip(valid_indices, summaries):
            result[idx] = summary
        
        return result
    
    def summarize_all(self, texts):
        """Process all texts in batches"""
        all_summaries = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Summarizing"):
            batch = texts[i:i + self.batch_size]
            summaries = self.summarize_batch(batch)
            all_summaries.extend(summaries)
        return all_summaries
# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    print("=" * 70)
    print("Google News Summarizer (Optimized for Colab)")
    print("=" * 70)
    
    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    START_DATE = datetime(2025, 4, 2)
    df = pd.read_csv('/kaggle/input/afterjuly18/filtered_after_july_15 - filtered_after_july_15.csv')
    print(f"Loaded {len(df)} articles")
    # df=df.head(5)
    
    # Step 1: Get actual URLs (reuses browser)
    print("\n[1/3] Fetching actual URLs...")
    start_time = time.time()
    df['actual_link'] = get_all_urls(df)
    print(f"Completed in {time.time() - start_time:.1f}s")
    
    # Step 2: Extract text (parallel-friendly)
    print("\n[2/3] Extracting text...")
    start_time = time.time()
    df['full_text'] = df['actual_link'].apply(extract_text)
    print(f"Completed in {time.time() - start_time:.1f}s")
    
    # # Step 3: Summarize (batched on GPU)
    # print("\n[3/3] Generating summaries...")
    # start_time = time.time()
    # summarizer = BatchSummarizer(batch_size=16)# Larger batch on GPU
    # df['summary'] = summarizer.summarize_all(df['full_text'].tolist())
    # print(f"Completed in {time.time() - start_time:.1f}s")
    
    # Save results
    df.to_csv('/kaggle/working/skye_after_july15.csv')
    
    # print("\n" + "=" * 70)
    # print(f"Total articles: {df.shape[0]}")
    # print(f"Summaries produced: {df['summary'].notna().sum()}")
    # print(f"Success rate: {df['summary'].notna().sum() / len(df) * 100:.1f}%")
    # print("=" * 70)