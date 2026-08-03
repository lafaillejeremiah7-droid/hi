"""Trade data aggregation module.

Aggregates individual Databento trade records from quarterly parquet files
into 5-minute OHLCV bars with REAL order flow data (bid/ask volume split).

Convention (Databento):
    - side='B' = BUYER aggressor (market buy lifting the offer) = bullish
    - side='A' = SELLER aggressor (market sell hitting the bid) = bearish
    - delta = bid_volume - ask_volume (positive = net buying = bullish)

Also generates footprint-level data (volume by price level per bar)
for advanced order flow analysis (absorption, stacked imbalances).
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def get_trade_files(trades_dir: str = "data/trades") -> list[Path]:
    """Get sorted list of quarterly trade parquet files.

    Args:
        trades_dir: Directory containing NQ_trades_YYYY_QN.parquet files.

    Returns:
        Sorted list of Path objects for all trade files.
    """
    pattern = str(Path(trades_dir) / "NQ_trades_*.parquet")
    files = sorted(glob.glob(pattern))
    return [Path(f) for f in files]


def aggregate_quarter(filepath: Path) -> pd.DataFrame:
    """Aggregate a single quarter's trade data into 5-min bars.

    Loads raw trades, filters to B/A sides, computes OHLCV plus
    bid_volume, ask_volume, delta, trade_count, avg_trade_size.

    Args:
        filepath: Path to quarterly parquet file.

    Returns:
        DataFrame with 5-min bars indexed by timestamp (US/Eastern).
    """
    df = pd.read_parquet(filepath, columns=["ts_event", "side", "price", "size"])

    # Cast size to int64 to avoid overflow in arithmetic
    df["size"] = df["size"].astype(np.int64)

    # Filter to only aggressor-classified trades (B and A)
    df = df[df["side"].isin(["B", "A"])].copy()

    if df.empty:
        return pd.DataFrame()

    # Use ts_event as the trade timestamp, convert to Eastern
    df = df.set_index("ts_event")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")

    # Compute buy/sell volume columns
    df["buy_volume"] = np.where(df["side"] == "B", df["size"], 0).astype(np.int64)
    df["sell_volume"] = np.where(df["side"] == "A", df["size"], 0).astype(np.int64)

    # Resample to 5-minute bars
    bars = df.resample("5min").agg(
        {
            "price": ["first", "max", "min", "last", "count"],
            "size": "sum",
            "buy_volume": "sum",
            "sell_volume": "sum",
        }
    )

    # Flatten multi-level columns
    bars.columns = [
        "open", "high", "low", "close", "trade_count",
        "volume", "bid_volume", "ask_volume",
    ]

    # Drop bars with no trades
    bars = bars.dropna(subset=["open"])
    bars = bars[bars["volume"] > 0]

    # Compute derived columns
    bars["delta"] = bars["bid_volume"] - bars["ask_volume"]
    bars["avg_trade_size"] = np.where(
        bars["trade_count"] > 0,
        bars["volume"] / bars["trade_count"],
        0.0,
    )

    # Ensure integer types for volume columns
    for col in ["volume", "bid_volume", "ask_volume", "delta", "trade_count"]:
        bars[col] = bars[col].astype(np.int64)

    return bars


def aggregate_all_trades(
    trades_dir: str = "data/trades",
    output_file: str = "data/NQ_5min_real_orderflow.parquet",
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate all quarterly trade files into a single 5-min bar dataset.

    Args:
        trades_dir: Directory containing quarterly parquet files.
        output_file: Path to save aggregated bars.
        verbose: Print progress messages.

    Returns:
        DataFrame with all 5-min bars.
    """
    files = get_trade_files(trades_dir)
    if not files:
        raise FileNotFoundError(
            f"No trade files found in {trades_dir}. "
            "Expected files like NQ_trades_2021_Q1.parquet"
        )

    all_bars = []
    for filepath in files:
        if verbose:
            print(f"  Processing {filepath.name}...")
        bars = aggregate_quarter(filepath)
        if not bars.empty:
            all_bars.append(bars)
            if verbose:
                print(f"    -> {len(bars):,} bars")

    if not all_bars:
        raise RuntimeError("No bars produced from trade files")

    # Concatenate and sort
    result = pd.concat(all_bars, axis=0)
    result = result.sort_index()

    # Remove any duplicate timestamps (shouldn't happen but be safe)
    result = result[~result.index.duplicated(keep="first")]

    # Save to parquet
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)

    if verbose:
        print(f"  Saved {len(result):,} bars to {output_file}")
        print(f"  Date range: {result.index[0]} to {result.index[-1]}")

    return result


def aggregate_footprint(
    trades_dir: str = "data/trades",
    output_file: str = "data/NQ_5min_footprint.parquet",
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate trades into footprint data (volume by price level per bar).

    Each row represents a price level within a 5-min bar, showing how
    much volume traded on the bid vs ask side at that level.

    This enables:
    - REAL absorption detection (heavy volume on BOTH sides at same level)
    - REAL imbalance detection (3x ratio between sides at consecutive levels)
    - Failed auction detection (no bid volume at high tick)
    - Trapped traders (heavy volume at bar extremes)

    Args:
        trades_dir: Directory containing quarterly parquet files.
        output_file: Path to save footprint data.
        verbose: Print progress messages.

    Returns:
        DataFrame with columns: bar_timestamp, price_level,
        bid_volume_at_level, ask_volume_at_level.
    """
    files = get_trade_files(trades_dir)
    if not files:
        raise FileNotFoundError(f"No trade files found in {trades_dir}")

    all_footprints = []
    for filepath in files:
        if verbose:
            print(f"  Processing footprint for {filepath.name}...")
        fp = _aggregate_quarter_footprint(filepath)
        if not fp.empty:
            all_footprints.append(fp)
            if verbose:
                print(f"    -> {len(fp):,} price-level rows")

    if not all_footprints:
        raise RuntimeError("No footprint data produced from trade files")

    result = pd.concat(all_footprints, axis=0)
    result = result.sort_values(["bar_timestamp", "price_level"])
    result = result.reset_index(drop=True)

    # Save to parquet
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)

    if verbose:
        print(f"  Saved {len(result):,} footprint rows to {output_file}")

    return result


def _aggregate_quarter_footprint(filepath: Path) -> pd.DataFrame:
    """Aggregate a single quarter into footprint data.

    Groups trades by 5-min bar and price level, computing bid/ask
    volume at each level.

    Args:
        filepath: Path to quarterly parquet file.

    Returns:
        DataFrame with bar_timestamp, price_level, bid_volume_at_level,
        ask_volume_at_level columns.
    """
    df = pd.read_parquet(filepath, columns=["ts_event", "side", "price", "size"])

    # Cast size to int64
    df["size"] = df["size"].astype(np.int64)

    # Filter to aggressor trades
    df = df[df["side"].isin(["B", "A"])].copy()

    if df.empty:
        return pd.DataFrame()

    # Use ts_event as trade timestamp, convert to Eastern
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("US/Eastern")

    # Create 5-min bar timestamp (floor to 5-min boundary)
    df["bar_timestamp"] = df["ts_event"].dt.floor("5min")

    # Compute bid/ask volume at each price level per bar
    df["bid_vol"] = np.where(df["side"] == "B", df["size"], 0).astype(np.int64)
    df["ask_vol"] = np.where(df["side"] == "A", df["size"], 0).astype(np.int64)

    # Group by bar + price level
    footprint = (
        df.groupby(["bar_timestamp", "price"])
        .agg(
            bid_volume_at_level=("bid_vol", "sum"),
            ask_volume_at_level=("ask_vol", "sum"),
        )
        .reset_index()
    )

    footprint = footprint.rename(columns={"price": "price_level"})

    return footprint


def load_or_aggregate_bars(
    trades_dir: str = "data/trades",
    output_file: str = "data/NQ_5min_real_orderflow.parquet",
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load aggregated bars from cache or aggregate from raw trades.

    Args:
        trades_dir: Directory containing raw trade files.
        output_file: Path to cached aggregated bars.
        force: If True, re-aggregate even if cache exists.
        verbose: Print progress messages.

    Returns:
        DataFrame with 5-min bars including real order flow data.
    """
    output_path = Path(output_file)

    if output_path.exists() and not force:
        if verbose:
            print(f"  Loading cached bars from {output_file}")
        df = pd.read_parquet(output_path)
        if verbose:
            print(f"  Loaded {len(df):,} bars")
        return df

    if verbose:
        print("  Aggregating raw trade data into 5-min bars...")
    return aggregate_all_trades(trades_dir, output_file, verbose)


def load_or_aggregate_footprint(
    trades_dir: str = "data/trades",
    output_file: str = "data/NQ_5min_footprint.parquet",
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load footprint data from cache or aggregate from raw trades.

    Args:
        trades_dir: Directory containing raw trade files.
        output_file: Path to cached footprint data.
        force: If True, re-aggregate even if cache exists.
        verbose: Print progress messages.

    Returns:
        DataFrame with footprint data (bar_timestamp, price_level, volumes).
    """
    output_path = Path(output_file)

    if output_path.exists() and not force:
        if verbose:
            print(f"  Loading cached footprint from {output_file}")
        df = pd.read_parquet(output_path)
        if verbose:
            print(f"  Loaded {len(df):,} footprint rows")
        return df

    if verbose:
        print("  Aggregating raw trade data into footprint...")
    return aggregate_footprint(trades_dir, output_file, verbose)
