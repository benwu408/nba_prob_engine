"""
Paths and defaults for the data pipeline.

- PROJECT_ROOT, DATA_DIR, RAW_DIR, PARSED_DIR: directory layout
- DEFAULT_SEASON: season string for nba_api (e.g. "2023-24")
- REQUEST_DELAY_SEC: delay between API requests to avoid rate limits
"""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"

# Season format for nba_api: "2023-24" for 2023-24 season
DEFAULT_SEASON = "2023-24"

# Rate limiting: nba_api can throttle; small delay between requests (seconds)
REQUEST_DELAY_SEC = 0.6
