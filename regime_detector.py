"""Utilities for detecting market regimes based on ADX, ER, and VHF values in parquet datasets."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import pandas as pd
import numpy as np


def detect_columns(df: pd.DataFrame, pattern: str) -> List[str]:
    """Detect columns in DataFrame matching a regex pattern."""
    regex = re.compile(pattern)
    return [col for col in df.columns if regex.match(col)]


def compute_adx_regimes(
    df: pd.DataFrame,
    adx_cols: Iterable[str],
    threshold: float,
) -> pd.DataFrame:
    """Generate binary regime labels from ADX columns."""
    out = df.copy()
    cols = [col for col in dict.fromkeys(adx_cols)]  # preserve order, drop duplicates
    
    missing = [col for col in cols if col not in out.columns]
    if missing:
        raise ValueError(f"ADX Column(s) {missing} not found in input DataFrame")

    for col in cols:
        # If only one ADX col was passed originally, the old code named it 'regime'.
        # But for auto-detection/multi-mode, explicit names are safer.
        # We will stick to f"regime_{col}" for consistency in multi-mode.
        out[f"regime_{col}"] = (out[col] > threshold).astype(int)
    
    return out


def compute_er_regimes(
    df: pd.DataFrame,
    er_cols: Iterable[str],
    thresh_low: float = 0.2,
    thresh_high: float = 0.4,
) -> pd.DataFrame:
    """Generate regime labels based on ER indicators."""
    out = df.copy()
    cols = [col for col in dict.fromkeys(er_cols)]

    missing = [col for col in cols if col not in out.columns]
    if missing:
        raise ValueError(f"ER Column(s) {missing} not found in input DataFrame")

    for col in cols:
        conditions = [
            (out[col] < thresh_low),
            (out[col] > thresh_high)
        ]
        choices = [0, 1]
        out[f"regime_{col}"] = np.select(conditions, choices, default=-1)
    
    return out


def compute_vhf_regimes(
    df: pd.DataFrame,
    vhf_cols: Iterable[str],
    thresh_low: float = 0.25,
    thresh_high: float = 0.5,
) -> pd.DataFrame:
    """Generate regime labels based on VHF indicators."""
    out = df.copy()
    cols = [col for col in dict.fromkeys(vhf_cols)]

    missing = [col for col in cols if col not in out.columns]
    if missing:
        raise ValueError(f"VHF Column(s) {missing} not found in input DataFrame")

    for col in cols:
        conditions = [
            (out[col] < thresh_low),
            (out[col] > thresh_high)
        ]
        choices = [0, 1]
        out[f"regime_{col}"] = np.select(conditions, choices, default=-1)
    
    return out


def _infer_coin_timeframe(input_path: Path) -> Tuple[str, str]:
    """Infer coin symbol and timeframe from a filename like BTC_5m_....parquet."""
    parts = input_path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            "Filename must contain at least `<COIN>_<TIMEFRAME>_...` to infer coin/timeframe"
        )
    return parts[0], parts[1]


def _default_output_path(input_path: Path) -> Path:
    coin, timeframe = _infer_coin_timeframe(input_path)
    out_dir = Path("data/regime") / coin / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{input_path.stem}.parquet").replace("_features", "_regime")


def main():
    parser = argparse.ArgumentParser(description="Detect regimes from ADX, ER, and VHF columns in parquet data")
    parser.add_argument("--input", required=True, help="Input parquet file")
    
    # ADX args
    parser.add_argument("--adx", nargs="+", help="Explicit ADX column names")
    parser.add_argument("--adx-threshold", default=25.0, type=float, help="ADX threshold for bullish regime")

    # ER args
    parser.add_argument("--er", nargs="+", help="Explicit ER column names")
    parser.add_argument("--er-thresh-low", default=0.2, type=float)
    parser.add_argument("--er-thresh-high", default=0.4, type=float)

    # VHF args
    parser.add_argument("--vhf", nargs="+", help="Explicit VHF column names")
    parser.add_argument("--vhf-thresh-low", default=0.25, type=float)
    parser.add_argument("--vhf-thresh-high", default=0.5, type=float)

    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output parquet path. Directories are created automatically.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    
    # Auto-detect if not provided
    adx_cols = args.adx if args.adx else detect_columns(df, r"^adx_\d+$")
    er_cols = args.er if args.er else detect_columns(df, r"^er_\d+$")
    vhf_cols = args.vhf if args.vhf else detect_columns(df, r"^vhf_\d+$")

    print(f"Processing file: {input_path}")
    
    if adx_cols:
        print(f"  ADX columns: {adx_cols} (threshold={args.adx_threshold})")
        df = compute_adx_regimes(df, adx_cols, args.adx_threshold)
    
    if er_cols:
        print(f"  ER columns: {er_cols} (low={args.er_thresh_low}, high={args.er_thresh_high})")
        df = compute_er_regimes(df, er_cols, args.er_thresh_low, args.er_thresh_high)
        
    if vhf_cols:
        print(f"  VHF columns: {vhf_cols} (low={args.vhf_thresh_low}, high={args.vhf_thresh_high})")
        df = compute_vhf_regimes(df, vhf_cols, args.vhf_thresh_low, args.vhf_thresh_high)

    if not (adx_cols or er_cols or vhf_cols):
        print("Warning: No ADX, ER, or VHF columns found or specified. No regimes generated.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = _default_output_path(input_path)

    df.to_parquet(output_path, index=False)
    print(f"Saved regime features to {output_path}")


if __name__ == "__main__":
    main()
