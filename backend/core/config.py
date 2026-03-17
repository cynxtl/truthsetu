from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "TruthSetu"
    app_env: str = "development"
    app_port: int = 8000

    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Ollama fallback
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # Database
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "truthsetu"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Embeddings & FAISS
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "./data/faiss_index"

    # WhatsApp
    green_api_instance_id: str = ""
    green_api_token: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_deploy_channel: str = ""

    # Twitter
    twitter_bearer_token: str = ""

    # RSS Sources
    pib_rss: str = "https://pib.gov.in/RssMain.aspx?ModID=6&Lang=1"
    imd_rss: str = "https://mausam.imd.gov.in/imd_latest/contents/rss-feed.php"
    who_rss: str = "https://www.who.int/rss-feeds/news-releases.xml"
    ndma_rss: str = "https://ndma.gov.in/RSS/ndma.xml"
    hindu_rss: str = "https://www.thehindu.com/news/feeder/default.rss"
    ie_rss: str = "https://indianexpress.com/feed/"
    ndtv_rss: str = "https://feeds.feedburner.com/ndtvnews-top-stories"
    altnews_rss: str = "https://www.altnews.in/feed/"
    boom_rss: str = "https://www.boomlive.in/feed"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()