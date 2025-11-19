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
    "er": {"length": [10]},
    "vhf": {"length": [14, 28]},
    "cvi": {"length": [10]},
    "kama_volatility": {"length": [10]},
    "supertrend": {"length": [10], "multiplier": [2.0]},
    "stochrsi": {"length": [14], "rsi_length": [14], "k": [3], "d": [3]},
}


def _as_seq(x) -> list:
    return list(x) if isinstance(x, (list, tuple, range, pd.Index)) else [x]


def compute_er(df: pd.DataFrame, price_col: str = "close", period: int = 10) -> pd.Series:
    """
    Compute Perry Kaufman's Efficiency Ratio (ER).

    ER_n = |price_n - price_{n-period}| / sum_{i=n-period+1..n} |price_i - price_{i-1}|

    Returns a Series aligned to df.index; insufficient lookback yields NaN.
    """
    if price_col not in df.columns:
        raise KeyError(f"compute_er requires column '{price_col}'")
    if period <= 0:
        raise ValueError("period must be positive")

    price = df[price_col]
    net_change = price.diff(periods=period).abs()
    volatility = price.diff().abs().rolling(window=period).sum()

    er = net_change / volatility
    er.name = f"er_{period}"
    return er


def compute_vhf(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 28,
) -> pd.Series:
    """
    Compute Vertical Horizontal Filter (VHF).

    VHF_n = (max_high_{n-period+1..n} - min_low_{n-period+1..n}) /
            sum_{i=n-period+1..n} |close_i - close_{i-1}|

    Returns a Series aligned to df.index; insufficient lookback yields NaN.
    """
    for col, name in zip((high_col, low_col, close_col), ("high", "low", "close")):
        if col not in df.columns:
            raise KeyError(f"compute_vhf requires column '{col}' for {name}")
    if period <= 0:
        raise ValueError("period must be positive")

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    price_range = highest_high - lowest_low

    volatility = close.diff().abs().rolling(window=period, min_periods=period).sum()

    vhf = price_range / volatility
    vhf.name = f"vhf_{period}"
    return vhf


def compute_cvi(
    df: pd.DataFrame,
    close_col: str = "close",
    period: int = 10,
) -> pd.Series:
    """
    Compute Chande Volatility Index (CVI).

    numerator   = EMA(|close - close.shift(1)|, period)
    denominator = EMA(close.shift(1), period)
    CVI         = 100 * numerator / denominator

    Returns Series named cvi_{period} aligned to df.index; insufficient data -> NaN.
    """
    if close_col not in df.columns:
        raise KeyError(f"compute_cvi requires column '{close_col}'")
    if period <= 0:
        raise ValueError("period must be positive")

    close = df[close_col]
    abs_change = (close - close.shift(1)).abs()

    numerator = abs_change.ewm(span=period, adjust=False).mean()
    denominator = close.shift(1).ewm(span=period, adjust=False).mean()

    cvi = 100 * numerator / denominator
    cvi = cvi.where(denominator != 0)
    cvi.name = f"cvi_{period}"
    return cvi


def compute_kama_volatility(
    df: pd.DataFrame,
    close_col: str = "close",
    period: int = 10,
) -> pd.Series:
    """
    Compute KAMA-based volatility (Perry Kaufman's volatility component).

    Volatility is the sum of absolute price changes over a period:
    vol_n = Σ |close_i - close_{i-1}| for i in [n-period+1, n]

    Returns a Series aligned to df.index; insufficient lookback yields NaN.
    """
    if close_col not in df.columns:
        raise KeyError(f"compute_kama_volatility requires column '{close_col}'")
    if period <= 0:
        raise ValueError("period must be positive")

    price = df[close_col]
    price_changes = price.diff().abs()
    volatility = price_changes.rolling(window=period, min_periods=period).sum()

    volatility.name = f"kama_vol_{period}"
    return volatility


def compute_supertrend(
    df: pd.DataFrame,
    atr_len: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Compute SuperTrend indicator.

    Returns a DataFrame with columns:
      supertrend      (the SuperTrend line)
      supertrend_dir  (trend direction: 1 for bullish, -1 for bearish)
    
    Aligned to df.index.
    """
    _require_columns(df, ["high", "low", "close"], "SuperTrend")
    
    # pandas_ta.supertrend returns a DataFrame with columns like:
    # SUPERT_{length}_{multiplier} (trend line)
    # SUPERTd_{length}_{multiplier} (direction: 1 or -1)
    # SUPERTl_{length}_{multiplier} (long trend)
    # SUPERTs_{length}_{multiplier} (short trend)
    st = ta.supertrend(
        high=df["high"], 
        low=df["low"], 
        close=df["close"], 
        length=atr_len, 
        multiplier=multiplier
    )
    
    # Identify the correct columns dynamically or by construction
    # The default naming convention in pandas_ta is:
    # SUPERT_{length}_{multiplier} -> value
    # SUPERTd_{length}_{multiplier} -> direction
    
    # We can just look for the columns that start with SUPERT and SUPERTd
    # But to be safe and precise given we know the params:
    # Note: pandas_ta might format floats with different precision in column names, 
    # so it's safer to grab by position or partial match if we are sure of the return structure.
    # ta.supertrend returns 4 columns.
    # Column 0: Trend Line (SUPERT_...)
    # Column 1: Direction (SUPERTd_...)
    # Column 2: Long Line
    # Column 3: Short Line
    
    if st is None or st.empty:
        # Fallback for insufficient data
        return pd.DataFrame(
            {"supertrend": [float("nan")] * len(df), "supertrend_dir": [float("nan")] * len(df)},
            index=df.index
        )

    out = pd.DataFrame(index=df.index)
    out["supertrend"] = st.iloc[:, 0]
    out["supertrend_dir"] = st.iloc[:, 1]
    
    return out


def compute_stochrsi(
    df: pd.DataFrame,
    rsi_len: int = 14,
    stoch_len: int = 14,
    k_len: int = 3,
    d_len: int = 3,
) -> pd.DataFrame:
    """
    Compute Stochastic RSI.

    Returns a DataFrame with columns:
      stochrsi_k
      stochrsi_d
    
    Aligned to df.index.
    """
    _require_columns(df, ["close"], "StochRSI")
    
    # pandas_ta.stochrsi returns a DataFrame with columns like:
    # STOCHRSIk_{length}_{rsi_length}_{k}_{d}
    # STOCHRSId_{length}_{rsi_length}_{k}_{d}
    stoch = ta.stochrsi(
        close=df["close"],
        length=stoch_len,
        rsi_length=rsi_len,
        k=k_len,
        d=d_len
    )
    
    if stoch is None or stoch.empty:
        return pd.DataFrame(
            {"stochrsi_k": [float("nan")] * len(df), "stochrsi_d": [float("nan")] * len(df)},
            index=df.index
        )

    out = pd.DataFrame(index=df.index)
    # pandas_ta returns K then D usually.
    # Column 0: K
    # Column 1: D
    out["stochrsi_k"] = stoch.iloc[:, 0]
    out["stochrsi_d"] = stoch.iloc[:, 1]
    
    return out


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

    # Efficiency Ratio (Perry Kaufman)
    if "er" in params:
        _require_columns(out, ["close"], "ER")
        for length in [int(x) for x in _as_seq(params["er"].get("length", 10))]:
            out[f"er_{length}"] = compute_er(out, price_col="close", period=int(length))

    # Vertical Horizontal Filter
    if "vhf" in params:
        _require_columns(out, ["high", "low", "close"], "VHF")
        for length in [int(x) for x in _as_seq(params["vhf"].get("length", 28))]:
            out[f"vhf_{length}"] = compute_vhf(out, period=int(length))

    # Chande Volatility Index
    if "cvi" in params:
        _require_columns(out, ["close"], "CVI")
        for length in [int(x) for x in _as_seq(params["cvi"].get("length", 10))]:
            out[f"cvi_{length}"] = compute_cvi(out, period=int(length))

    # KAMA-based Volatility
    if "kama_volatility" in params:
        _require_columns(out, ["close"], "KAMA Volatility")
        for length in [int(x) for x in _as_seq(params["kama_volatility"].get("length", 10))]:
            out[f"kama_vol_{length}"] = compute_kama_volatility(out, close_col="close", period=int(length))

    # SuperTrend
    if "supertrend" in params:
        # _require_columns checked inside compute_supertrend
        lengths = [int(x) for x in _as_seq(params["supertrend"].get("length", 10))]
        multipliers = [float(x) for x in _as_seq(params["supertrend"].get("multiplier", 2.0))]
        
        for length, mult in product(lengths, multipliers):
            st_df = compute_supertrend(out, atr_len=length, multiplier=mult)
            out[f"supertrend_{length}_{mult}"] = st_df["supertrend"]
            out[f"supertrend_dir_{length}_{mult}"] = st_df["supertrend_dir"]

    # StochRSI
    if "stochrsi" in params:
        _require_columns(out, ["close"], "StochRSI")
        # We iterate over 'length' (stoch_len) and 'rsi_length' (rsi_len) and k, d
        # Default params structure: {"length": [14], "rsi_length": [14], "k": [3], "d": [3]}
        
        stoch_lens = [int(x) for x in _as_seq(params["stochrsi"].get("length", 14))]
        rsi_lens = [int(x) for x in _as_seq(params["stochrsi"].get("rsi_length", 14))]
        ks = [int(x) for x in _as_seq(params["stochrsi"].get("k", 3))]
        ds = [int(x) for x in _as_seq(params["stochrsi"].get("d", 3))]
        
        for sl, rl, k, d in product(stoch_lens, rsi_lens, ks, ds):
            s_df = compute_stochrsi(out, rsi_len=rl, stoch_len=sl, k_len=k, d_len=d)
            # suffix: _<stoch_len>_<rsi_len>_<k>_<d>
            suffix = f"{sl}_{rl}_{k}_{d}"
            out[f"stochrsi_k_{suffix}"] = s_df["stochrsi_k"]
            out[f"stochrsi_d_{suffix}"] = s_df["stochrsi_d"]

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
