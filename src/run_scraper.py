from yfinance_scraper import YahooScraper
from gnews_scraper import GNewsScraper

ENTITIES = {
    'BA': ['Boeing', 'BA'],
    'RDDT': ['Reddit', 'RDDT'],
    'AAPL': ['Apple', 'AAPL'],
    'NVO': ['Novo Nordisk', 'NVO'],
    'DJT': ['Trump Media', 'DJT', 'Truth Social'],
    'TSN': ['Tyson Foods', 'TSN'],
    'NVDA': ['Nvidia', 'NVDA'],
    'NYCB': ['NYCB', 'New York Community Bank'],
    'BK': ['Bank of New York Mellon Corp', 'BNY'],
    'CMG': ['Chipotle', 'CMG'],
    'TSLA': ['Tesla', 'TSLA']
}

MACRO_TOPICS = ['interest rates', 'inflation', 'volatility']

def scrape_all():
    try:
        yfinance = YahooScraper(list(ENTITIES.keys()))
        yfinance.fetch().save()
    except Exception as e:
        print(f"Critical Failure in Yahoo Pipeline: {e}")

    try:
        gnews = GNewsScraper(entities=ENTITIES, topics=MACRO_TOPICS)
        gnews.fetch().save()
    except Exception as e:
        print(f"Critical Failure in Gnews Pipeline: {e}")

if __name__ == "__main__":
    scrape_all()