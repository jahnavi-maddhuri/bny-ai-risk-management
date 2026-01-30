import hashlib
import re
from datetime import datetime
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from abc import ABC, abstractmethod
from database import db
from dotenv import load_dotenv

load_dotenv()

class BaseScraper(ABC):
    def __init__(self):
        self.run_ts = datetime.now().isoformat()
        self.batch = []
        self.db = db

    def clean_url(self, url):
        if not url: return ""
        try:
            p = urlparse(url)
            q = [(k, v) for k, v in parse_qsl(p.query) if not re.match(r'^(utm_|gclid|ref)', k, re.I)]
            return urlunparse((p.scheme, p.netloc, p.path, "", urlencode(q), ""))
        except Exception as e:
            print(f"URL Cleaning Error: {e}")
            return url

    def hash_url(self, url):
        """Creates the unique ID for our database upsert logic."""
        return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()

    def save(self):
        """
        Sends all results to Supabase in a single batch.
        Uses 'on_conflict' to skip duplicates based on the URL hash.
        """
        if not self.batch:
            print(f"[{self.__class__.__name__}] Nothing to save.")
            return self
        
        try:
            print(f"[{self.__class__.__name__}] Saving {len(self.batch)} items to DB...")
            self.db.upsert_raw_news(self.batch)
            self.batch = [] 
        except Exception as e:
            print(f"!! [DATABASE ERROR] Failed to save {self.__class__.__name__} batch: {e}")
        
        return self
    @abstractmethod
    def fetch(self):
        pass
