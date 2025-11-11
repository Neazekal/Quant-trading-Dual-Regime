"""Utility module for cleaning 5-minute crypto data sets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


CONFIG: Dict[str, str] = {
    "timezone": "Asia/Ho_Chi_Minh",
    "target_tz": "UTC",
    "freq": "5T",
}

# Update these paths to match your environment before running the script.
INPUT_CSV = Path("data/DOGEUSDT_5m_merged_20250101_20250110.csv")
OUTPUT_DIR = Path("data/DOGEUSDT_5m_cleaned")

EIGHT_HOURS = pd.Timedelta("8h")


def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Read the raw CSV and return a timezone-aware DataFrame indexed by UTC timestamps."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    source_tz = CONFIG["timezone"]
    target_tz = CONFIG["target_tz"]

    if df.index.tz is None:
        df.index = df.index.tz_localize(source_tz)
    else:
        df.index = df.index.tz_convert(source_tz)

    df.index = df.index.tz_convert(target_tz)
    return df

