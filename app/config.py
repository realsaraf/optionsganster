"""
OptionsGangster Configuration
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
    OPENAI_API_KEY: str = ""   # optional — enables LLM narrative layer

    # Firebase / Google Auth
    FIREBASE_PROJECT_ID: str = ""
    FIREBASE_API_KEY: str = ""
    FIREBASE_AUTH_DOMAIN: str = ""
    FIREBASE_APP_ID: str = ""
    FIREBASE_MESSAGING_SENDER_ID: str = ""
    FIREBASE_STORAGE_BUCKET: str = ""
    FIREBASE_MEASUREMENT_ID: str = ""

    # MongoDB Atlas
    MONGO_URI: str = ""
    MONGO_DB: str = ""

    # Twilio SMS notifications
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_SMS_FROM: str = ""
    TWILIO_SMS_TO: str = ""
    TWILIO_SMS_ALERT_TYPES: str = "actionable"

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
