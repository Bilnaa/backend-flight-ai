"""
Configuration de l'application
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration de l'application"""

    # Base de données
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/flightdb")

    # Ray
    RAY_ADDRESS: str = os.getenv("RAY_ADDRESS", "auto")

    # Modèle ML
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/modele.pkl")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: str = os.getenv("API_PORT", "8000")

    # Environnement
    ENV: str = os.getenv("ENV", "production")


settings = Settings()
