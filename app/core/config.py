import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or with defaults
    """
    # API server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # API keys
    OPEN_API_KEY: str = os.getenv("OPEN_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Caching settings
    USE_GPTCACHE: bool = os.getenv("USE_GPTCACHE", "true").lower() == "true"

    # Zilliz/Milvus settings
    ZILLIZ_URI: str = os.getenv("ZILLIZ_URI", "")
    ZILLIZ_TOKEN: str = os.getenv("ZILLIZ_TOKEN", "")
    WIPE_MILVUS_ON_START: bool = os.getenv("WIPE_MILVUS_ON_START", "false").lower() == "true"

    # Model settings
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.0-pro")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    # Local encoder backend: bert, roberta or distilbert.
    ENCODER_MODEL: str = os.getenv("ENCODER_MODEL", "bert")
    SKILL_CATALOGUE: str = os.getenv("SKILL_CATALOGUE", "data/skills.txt")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings():
    """
    Get settings with caching to avoid reloading
    """
    return Settings()
