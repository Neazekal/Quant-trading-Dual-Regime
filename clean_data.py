import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple

# ==================== CONFIG ====================
CONFIG: Dict[str, str] = {
    "timezone": "Asia/Ho_Chi_Minh",
    "target_tz": "UTC",
    "freq": "5T",
}

EIGHT_HOURS = pd.Timedelta("8h")


# ==================== CORE FUNCTIONS ====================
def _normalize_freq(freq: str) -> str:
    if not freq:
        raise ValueError("CONFIG['freq'] must be a non-empty string.")
    freq = str(freq).strip()
    pattern = re.fullmatch(r"(\d*)([A-Za-z]+)", freq)
    if not pattern:
        return freq.lower()

    multiplier, unit = pattern.groups()
    unit_lower = unit.lower()
    alias_map = {"t": "min", "m": "min", "min": "min", "h": "h", "hr": "h", "hrs": "h"}
    normalized_unit = alias_map.get(unit_lower, unit_lower)
    return f"{multiplier}{normalized_unit}"


def load_data(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include 'timestamp' column.")

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
    df = df.copy()
    df["tradable_flag"] = df["volume"].fillna(0).ne(0)

    close = df["close"]
    returns = close.pct_change()

    price_range = df["high"] - df["low"]
    close_array = close.to_numpy(dtype=float)
    range_array = price_range.to_numpy(dtype=float)
    range_ratio = np.divide(range_array, close_array, out=np.full_like(range_array, np.nan), where=close_array != 0)
    range_ratio_series = pd.Series(range_ratio, index=df.index)

    outlier_condition = returns.abs().gt(0.1) | range_ratio_series.gt(0.15)
    outlier_condition = outlier_condition.fillna(False)
    if not outlier_condition.empty:
        outlier_condition.iloc[0] = False

    df["outlier_flag"] = outlier_condition
    return df


def save_data(df: pd.DataFrame, out_prefix: Path | str) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "open", "high", "low", "close", "volume",
        "funding_rate", "tradable_flag", "outlier_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    export_df = df[required_cols].copy()
    export_df.index.name = "timestamp"

    export_df.to_parquet(f"{out_prefix}.parquet")
    # export_df.to_csv(f"{out_prefix}.csv", index_label="timestamp")
    
def auto_output_path(input_file: Path) -> Path:
    stem = input_file.stem  # DOGE_5m_20240101_20251112
    parts = stem.split("_")

    if len(parts) < 4:
        raise ValueError(f"Invalid filename format: {input_file.name}")

    symbol = parts[0]           
    timeframe = parts[1]        

    outdir = Path("data") / "cleaned_data" / symbol / timeframe
    outdir.mkdir(parents=True, exist_ok=True)

    return outdir / (stem + "_clean")



# ==================== ARGPARSE MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Clean merged OHLCV + funding data")
    parser.add_argument("--input", required=True, help="Path to merged CSV input file")
    parser.add_argument("--outdir", default=None, help="Optional output directory")
    parser.add_argument("--freq", default="5T", help="Resample frequency")
    args = parser.parse_args()

    CONFIG["freq"] = args.freq

    input_path = Path(args.input)

    # ===== AUTO OUTPUT FOLDER =====
    if args.outdir is None:
        out_prefix = auto_output_path(input_path)
    else:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_prefix = outdir / input_path.stem

    df_raw = load_data(input_path)
    df_clean, stats = clean_data(df_raw)
    df_enriched = add_flags(df_clean)

    save_data(df_enriched, out_prefix)

    print(f"Saved:")
    # print(f"  {out_prefix}_clean.csv")
    print(f"  {out_prefix}_clean.parquet")


if __name__ == "__main__":
    main()
