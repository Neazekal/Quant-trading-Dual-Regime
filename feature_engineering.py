"""Feature engineering utilities for technical indicators using pandas_ta."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, Sequence
from itertools import product
import argparse

import pandas as pd
import pandas_ta as ta


# -------- Default indicator parameters --------
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "rsi": {"length": [7, 14, 21]},
    "bbands": {"length": [20, 40], "std": [1.5, 2.0, 2.5]},
    "ema_fast": {"length": [10, 20]},
    "ema_slow": {"length": [50, 100]},
    "adx": {"length": [14, 28]},
    "atr": {"length": [14]},
}


def _as_seq(x) -> list:
    return list(x) if isinstance(x, (list, tuple, range, pd.Index)) else [x]


# -------- Small utilities --------
def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} requires columns {list(cols)} but missing {missing}")


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is a 'timestamp' column for downstream compatibility."""
    out = df.copy()
    if "timestamp" not in out.columns:
        # common parquet exports keep the index; normalize to 'timestamp'
        if "index" in out.columns:
            out = out.rename(columns={"index": "timestamp"})
        elif out.index.name:
            out = out.rename_axis("timestamp").reset_index()
        else:
            out = out.reset_index(drop=False).rename(columns={"index": "timestamp"})
    return out


def _auto_output_prefix(in_path: Path) -> Path:
    """
    Automatically derive the output prefix: data/<COIN>/<TIMEFRAME>/<stem>_features
    Assumes filenames follow COIN_TIMEFRAME_... with at least two underscore-separated parts.
    """
    stem = in_path.stem  # e.g. DOGE_5m_20240101_20251112_clean
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer symbol/timeframe from filename: {in_path.name}")

    symbol = parts[0]      # DOGE
    timeframe = parts[1]   # 5m

    outdir = Path("data/featured_data") / symbol / timeframe
    outdir.mkdir(parents=True, exist_ok=True)

    return outdir / stem  # _features suffix added during save step


# -------- Core feature generation (grid-aware) --------
def generate_features(
    df: pd.DataFrame,
    params: Dict[str, Dict[str, Any]],
    *,
    shift: int = 0,          # shift > 0 to avoid look-ahead when training ML models
) -> pd.DataFrame:
    """
    Return a new DataFrame with engineered features.
    Each parameter accepts scalars or iterable ranges.
    """
    out = df.copy()

    # ADX
    if "adx" in params:
        _require_columns(out, ["high", "low", "close"], "ADX")
        for length in _as_seq(params["adx"].get("length", 14)):
            length = int(length)
            adx_df = ta.adx(high=out["high"], low=out["low"], close=out["close"], length=length)
            out[f"adx_{length}"] = adx_df[f"ADX_{length}"]

    # RSI
    if "rsi" in params:
        _require_columns(out, ["close"], "RSI")
        for length in _as_seq(params["rsi"].get("length", 14)):
            length = int(length)
            out[f"rsi_{length}"] = ta.rsi(close=out["close"], length=length)

    # Bollinger Bands: iterate every (length, std) combination
    if "bbands" in params:
        _require_columns(out, ["close"], "BBANDS")
        lengths = [int(x) for x in _as_seq(params["bbands"].get("length", 20))]
        stds = [float(x) for x in _as_seq(params["bbands"].get("std", 2))]
        for length, std in product(lengths, stds):
            bb = ta.bbands(close=out["close"], length=length, std=std)
            prefix_map = {c.split("_", 1)[0]: c for c in bb.columns}
            for need in ("BBU", "BBM", "BBL"):
                if need not in prefix_map:
                    raise KeyError(f"BBANDS missing {need}. Columns: {list(bb.columns)}")
            out[f"bb_upper_{length}_{std}"]  = bb[prefix_map["BBU"]]
            out[f"bb_middle_{length}_{std}"] = bb[prefix_map["BBM"]]
            out[f"bb_lower_{length}_{std}"]  = bb[prefix_map["BBL"]]

    # EMA fast
    if "ema_fast" in params:
        _require_columns(out, ["close"], "EMA(fast)")
        for fast_len in [int(x) for x in _as_seq(params["ema_fast"].get("length", 20))]:
            col_fast = f"ema_fast_{fast_len}"
            out[col_fast] = ta.ema(close=out["close"], length=fast_len)

    # EMA slow
    if "ema_slow" in params:
        _require_columns(out, ["close"], "EMA(slow)")
        for slow_len in [int(x) for x in _as_seq(params["ema_slow"].get("length", 50))]:
            col_slow = f"ema_slow_{slow_len}"
            reused = f"ema_fast_{slow_len}"
            out[col_slow] = out[reused] if reused in out.columns else ta.ema(out["close"], length=slow_len)

    # ATR
    if "atr" in params:
        _require_columns(out, ["high", "low", "close"], "ATR")
        for length in [int(x) for x in _as_seq(params["atr"].get("length", 14))]:
            out[f"atr_{length}"] = ta.atr(high=out["high"], low=out["low"], close=out["close"], length=length)

    # optional anti-lookahead
    if shift:
        feat_cols = [c for c in out.columns if c not in df.columns]
        out[feat_cols] = out[feat_cols].shift(shift)

    return out


# -------- CLI entry point --------
def main():
    p = argparse.ArgumentParser(description="Generate technical indicator features")
    p.add_argument("--input", required=True, help="Parquet")
    p.add_argument("--outdir", default=None, help="Optional output base directory")
    p.add_argument("--shift", type=int, default=1, help="Shift features to avoid look-ahead")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # read the file
    df = pd.read_parquet(in_path)

    df = _ensure_timestamp(df)

    df_feat = generate_features(df, DEFAULT_PARAMS, shift=args.shift).dropna()

    # determine the output prefix
    if args.outdir is None:
        out_prefix = _auto_output_prefix(in_path)
    else:
        base = Path(args.outdir)
        base.mkdir(parents=True, exist_ok=True)
        out_prefix = base / in_path.stem
    
    out_prefix = str(out_prefix).replace("_clean", "_features")

    df_feat.to_parquet(f"{out_prefix}.parquet", index=False)
    # # df_feat.to_csv(f"{out_prefix}.csv", index=False)
    print(f"Saved to {out_prefix}.parquet")


if __name__ == "__main__":
    main()
