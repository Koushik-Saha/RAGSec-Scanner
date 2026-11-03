"""Configuration, read from environment variables only.

This is the single module that reads os.environ. Everything else receives a
Settings object (or values from it).
"""

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ragsec"

DEFAULT_DATA_DIR = "./data"
DEFAULT_CHROMA_DIR = "./chroma"
DEFAULT_REPORTS_DIR = "./reports"


class Settings(BaseModel):
    anthropic_api_key: SecretStr | None = None
    model: str | None = None
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    chroma_dir: Path = Path(DEFAULT_CHROMA_DIR)
    reports_dir: Path = Path(DEFAULT_REPORTS_DIR)
    embedding_model: str = EMBEDDING_MODEL
    collection_name: str = COLLECTION_NAME


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env`, or from the process environment (plus .env) if None.

    Values already set in the environment take precedence over .env.
    """
    if env is None:
        load_dotenv(override=False)
        env = os.environ
    return Settings(
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
        model=env.get("RAGSEC_MODEL") or None,
        data_dir=Path(env.get("RAGSEC_DATA_DIR") or DEFAULT_DATA_DIR),
        chroma_dir=Path(env.get("RAGSEC_CHROMA_DIR") or DEFAULT_CHROMA_DIR),
        reports_dir=Path(env.get("RAGSEC_REPORTS_DIR") or DEFAULT_REPORTS_DIR),
    )
