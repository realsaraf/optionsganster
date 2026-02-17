"""
OptionsGanster Configuration
Uses Pydantic BaseSettings for validated, typed config with .env auto-loading.
"""
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    POLYGON_API_KEY: str = ""
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    # Cache TTLs (seconds)
    CACHE_TTL_EXPIRATIONS: int = 3600  # 1 hour – expiry dates rarely change
    CACHE_TTL_STRIKES: int = 3600      # 1 hour – strikes rarely change
    CACHE_TTL_OHLCV: int = 60  # 1 min
    CACHE_TTL_PRICES: int = 30  # 30 sec

    model_config = {
        "env_file": str(Path(__file__).parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
