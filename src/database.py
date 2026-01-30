import os
from supabase import create_client, Client
from constant import Constant
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        self.client: Client = create_client(url, key)

    def upsert_raw_news(self, batch):
        if not batch: return
        return self.client.table(Constant.RAW_NEWS).upsert(batch, on_conflict=Constant.URL_HASH).execute()


db = DatabaseManager()