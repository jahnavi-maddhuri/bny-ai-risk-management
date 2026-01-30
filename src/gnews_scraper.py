
from base_scraper import BaseScraper
from pygooglenews import GoogleNews
from constant import Constant


class GNewsScraper(BaseScraper):
    def __init__(self, entities, topics=None, period='1h', start_date=None, end_date=None):
        super().__init__()
        self.entities = entities or {}
        self.topics = topics or []
        self.period = period
        self.start_date = start_date
        self.end_date = end_date
        self.gn = GoogleNews(lang='en', country='US')

    def fetch(self):
        search_tasks = []
        
        for ticker, names in self.entities.items():
            query = f"({' OR '.join([f'\"{n}\"' for n in names])}) -gossip"
            search_tasks.append((query, ticker, Constant.TOPIC_COMPANY_NEWS))
            
        for topic in self.topics:
            search_tasks.append((f'"{topic}"', None, Constant.TOPIC_MACROTOPIC))

        for query_str, ticker, topic_label in search_tasks:
            try:
                print(f"GNews: Searching {topic_label} -> {query_str[:50]}...")
                
                if self.start_date and self.end_date:
                    res = self.gn.search(query_str, from_=self.start_date, to_=self.end_date)
                else:
                    res = self.gn.search(query_str, when=self.period)

                for entry in res.get("entries", []):
                    clean_url = self.clean_url(entry.get('link'))
                    self.batch.append({
                        Constant.URL_HASH: self.hash_url(clean_url),
                        Constant.TITLE: entry.get('title'),
                        Constant.URL: clean_url,
                        Constant.PUBLISHED_AT: entry.get('published'),
                        Constant.TICKER: ticker,     # Will be None for macro topics
                        Constant.TOPIC: topic_label, # 'company_news' or 'macro'
                        Constant.SOURCE: Constant.SOURCE_GNEWS
                    })
            except Exception as e:
                            print(f"!! [SCRAPE ERROR] Failed query '{query_str}': {e}")
                            continue 
            
        return self
                            
