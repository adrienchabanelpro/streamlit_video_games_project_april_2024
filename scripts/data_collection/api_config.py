"""API key management for data collection scripts.

All API keys are loaded from environment variables or a .env file.
Never hardcode keys in source files.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def get_rawg_api_key() -> str:
    """Return the RAWG API key from environment."""
    key = os.environ.get("RAWG_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "RAWG_API_KEY not set. Get a free key at https://rawg.io/apidocs "
            "and add it to your .env file."
        )
    return key


def get_igdb_credentials() -> tuple[str, str]:
    """Return (client_id, client_secret) for IGDB/Twitch API."""
    client_id = os.environ.get("TWITCH_CLIENT_ID", "")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET not set. "
            "Register at https://dev.twitch.tv/console and add to .env."
        )
    return client_id, client_secret


def get_igdb_access_token() -> str:
    """Obtain an OAuth2 access token for the IGDB API."""
    import requests

    client_id, client_secret = get_igdb_credentials()
    resp = requests.post(
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]
