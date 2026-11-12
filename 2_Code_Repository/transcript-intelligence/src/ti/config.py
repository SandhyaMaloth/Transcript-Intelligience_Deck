"""Paths and dataset-specific event dates, defined once for the whole package."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"
OUTPUT_DIR = ROOT / "outputs"
LABEL_CACHE = ROOT / "labels" / "llm_labels.json"
PROMPT_FILE = ROOT / "prompts" / "classify_call.md"

INTERNAL_DOMAIN = "aegiscloud.com"

# The two events that shape this dataset's timeline.
OUTAGE_DAY = pd.Timestamp("2026-03-10", tz="UTC")     # Detect ingestion outage
RECOVERY_START = pd.Timestamp("2026-03-24", tz="UTC")  # end of the outage-fallout fortnight
LAUNCH_DAY = pd.Timestamp("2026-04-07", tz="UTC")     # Comply v2 general availability
