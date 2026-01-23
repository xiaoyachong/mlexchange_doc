import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"

settings = Settings()
