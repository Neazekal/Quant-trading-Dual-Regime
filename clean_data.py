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


def _normalize_freq(freq: str) -> str:
    """Map deprecated pandas aliases to their future-safe equivalents."""
    if not freq:
        raise ValueError("CONFIG['freq'] must be a non-empty string.")

    freq = str(freq).strip()
    pattern = re.fullmatch(r"(\d*)([A-Za-z]+)", freq)
    if not pattern:
        return freq.lower()

    multiplier, unit = pattern.groups()
    unit_lower = unit.lower()
    alias_map = {
        "t": "min",
        "m": "min",
        "min": "min",
        "h": "h",
        "hr": "h",
        "hrs": "h",
    }
    normalized_unit = alias_map.get(unit_lower, unit_lower)
    return f"{multiplier}{normalized_unit}"


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


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicate timestamps, enforce fixed frequency, and tidy funding data."""
    if df.empty:
        raise ValueError("Input DataFrame is empty after loading.")

    df = df.sort_index()
    duplicate_mask = df.index.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        df = df[~duplicate_mask]

    df = df.rename(columns={"fundingRate": "funding_rate"})
    if "funding_rate" not in df.columns:
        df["funding_rate"] = np.nan

    freq = _normalize_freq(CONFIG["freq"])
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
        tz=df.index.tz,
    )

    df = df.reindex(full_index)
    df.index.name = "timestamp"

    freq_delta = pd.to_timedelta(freq)
    fill_limit = max(int(EIGHT_HOURS / freq_delta), 1)
    df["funding_rate"] = df["funding_rate"].ffill(limit=fill_limit)

    stats = {"duplicates_removed": duplicate_count}
    return df, stats


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute tradable and outlier flags."""
    df = df.copy()

    volume_series = df["volume"].fillna(0)
    df["tradable_flag"] = volume_series.ne(0)

    close = df["close"]
    returns = close.pct_change()

    price_range = df["high"] - df["low"]
    close_array = close.to_numpy(dtype=float)
    range_array = price_range.to_numpy(dtype=float)
    range_ratio = np.divide(
        range_array,
        close_array,
        out=np.full_like(range_array, np.nan),
        where=close_array != 0,
    )
    range_ratio_series = pd.Series(range_ratio, index=df.index)

    outlier_condition = returns.abs().gt(0.1) | range_ratio_series.gt(0.15)
    outlier_condition = outlier_condition.fillna(False)
    if not outlier_condition.empty:
        outlier_condition.iloc[0] = False
    df["outlier_flag"] = outlier_condition

    return df


def save_data(df: pd.DataFrame, out_prefix: Path | str) -> None:
    """Persist the cleaned dataset to Parquet and CSV files."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "tradable_flag",
        "outlier_flag",
    ]

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for export: {missing_cols}")

    export_df = df[columns].copy()
    export_df.index.name = "timestamp"

    export_df.to_parquet(f"{out_prefix}_clean.parquet")
    export_df.to_csv(f"{out_prefix}_clean.csv", index_label="timestamp")


if __name__ == "__main__":
    raw_df = load_data(INPUT_CSV)
    cleaned_df, stats = clean_data(raw_df)
    enriched_df = add_flags(cleaned_df)

    output_prefix = OUTPUT_DIR / INPUT_CSV.stem
    save_data(enriched_df, output_prefix)

    total_bars = len(enriched_df)
    zero_volume_count = int((enriched_df["volume"] == 0).sum())
    outlier_count = int(enriched_df["outlier_flag"].sum())

    print(f"Total bars: {total_bars}")
    print(f"Duplicates removed: {stats.get('duplicates_removed', 0)}")
    print(f"Zero-volume bars: {zero_volume_count}")
    print(f"Outlier bars: {outlier_count}")
