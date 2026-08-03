"""Data preprocessor module.

Computes order flow indicators and support/resistance levels.
When real order flow data is available (bid_volume, ask_volume, delta),
uses REAL signals instead of OHLCV proxies.

Handles both daily and intraday data with proper session boundaries
and daily resets for intraday metrics.

Optimized for large datasets (100K+ bars) using vectorized operations.
"""

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to data.

    Detects whether real order flow columns are present (bid_volume,
    ask_volume, delta) and uses them for accurate signals. Falls back
    to OHLCV proxies when real data is not available.

    Computes:
    - volume_delta: real delta if available, else proxy
    - cumulative_delta: running sum (daily reset for intraday)
    - relative_volume: volume / rolling mean for spike detection
    - vwap: volume-weighted average price (daily reset for intraday)
    - support/resistance levels
    - absorption_signal: bars where BOTH sides have heavy volume (real OF only)
    - imbalance_signal: bars where one side dominates heavily (real OF only)

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
            and optionally [bid_volume, ask_volume, delta, trade_count].

    Returns:
        DataFrame with additional computed columns.
    """
    df = df.copy()

    has_real_orderflow = all(
        col in df.columns for col in ["bid_volume", "ask_volume", "delta"]
    )

    if has_real_orderflow:
        # Use REAL order flow data
        df["volume_delta"] = df["delta"].astype(float)
    else:
        # Fall back to OHLCV proxy
        df["volume_delta"] = compute_volume_delta(df)

    # Compute cumulative delta (daily reset for intraday)
    df["cumulative_delta"] = compute_cumulative_delta(df)

    # Compute relative volume
    df["relative_volume"] = compute_relative_volume(df, window=20)

    # Compute VWAP (daily reset for intraday)
    df["vwap"] = compute_vwap(df)

    # Compute support/resistance levels
    df = compute_support_resistance(df)

    # For intraday data, add time-of-day relative volume
    if _is_intraday(df):
        df["relative_volume_tod"] = compute_relative_volume_tod(df)

    # Real order flow specific signals
    if has_real_orderflow:
        df["absorption_signal"] = compute_absorption_signal(df)
        df["imbalance_signal"] = compute_imbalance_signal(df)

    # Compute delta Z-score for dynamic exit management
    df["delta_zscore"] = compute_delta_zscore(df, window=50)

    return df


def _is_intraday(df: pd.DataFrame) -> bool:
    """Check if the data is intraday (multiple bars per day)."""
    if not hasattr(df.index, "date"):
        return False
    if len(df) < 2:
        return False
    dates = pd.Series(df.index.date, index=df.index)
    unique_dates = dates.nunique()
    return unique_dates < len(df) * 0.9


def compute_volume_delta(df: pd.DataFrame) -> pd.Series:
    """Compute volume delta as a proxy for buy/sell pressure.

    Used only when real order flow data is not available.

    Formula: volume_delta = volume * (close - open) / (high - low)
    When high == low (doji), delta is 0.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        Series of volume delta values.
    """
    bar_range = df["high"] - df["low"]
    bar_range = bar_range.replace(0, np.nan)
    delta = df["volume"] * (df["close"] - df["open"]) / bar_range
    return delta.fillna(0)


def compute_cumulative_delta(df: pd.DataFrame) -> pd.Series:
    """Compute cumulative delta with daily reset for intraday data.

    Uses real delta column if available, otherwise uses volume_delta proxy.
    Resets at the start of each trading day for intraday data.

    Args:
        df: DataFrame with 'volume_delta' column or OHLCV for computation.

    Returns:
        Series of cumulative delta values.
    """
    if "volume_delta" in df.columns:
        volume_delta = df["volume_delta"]
    else:
        volume_delta = compute_volume_delta(df)

    # Check if we have intraday data (multiple bars per day)
    if hasattr(df.index, "date"):
        dates = pd.Series(df.index.date, index=df.index)
        unique_dates = dates.nunique()
        if unique_dates < len(df) * 0.9:
            # Intraday: reset cumulative delta daily
            cumulative = volume_delta.groupby(dates).cumsum()
        else:
            cumulative = volume_delta.cumsum()
    else:
        cumulative = volume_delta.cumsum()

    return cumulative


def compute_relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute relative volume (volume / rolling average).

    Values > 1 indicate above-average volume.
    Values > 2 indicate significant volume spikes.

    Args:
        df: DataFrame with 'volume' column.
        window: Rolling window for average volume calculation.

    Returns:
        Series of relative volume values.
    """
    rolling_avg = df["volume"].rolling(window=window, min_periods=1).mean()
    rolling_avg = rolling_avg.replace(0, np.nan)
    rel_vol = df["volume"] / rolling_avg
    return rel_vol.fillna(1.0)


def compute_relative_volume_tod(df: pd.DataFrame, lookback_days: int = 10) -> pd.Series:
    """Compute relative volume compared to same time-of-day average.

    For intraday data, compares each bar's volume to the average volume
    at the same time of day over the past N days.

    Args:
        df: DataFrame with 'volume' column and DatetimeIndex.
        lookback_days: Number of days to average over.

    Returns:
        Series of time-of-day relative volume values.
    """
    if not hasattr(df.index, "time"):
        return pd.Series(1.0, index=df.index)

    time_key = df.index.strftime("%H:%M")
    result = pd.Series(1.0, index=df.index, dtype=float)

    for time_val, group in df.groupby(time_key):
        if len(group) <= 1:
            continue
        rolling_avg = group["volume"].rolling(window=lookback_days, min_periods=1).mean().shift(1)
        rolling_avg = rolling_avg.fillna(group["volume"].iloc[0])
        rolling_avg = rolling_avg.replace(0, 1)
        ratio = group["volume"] / rolling_avg
        result.loc[group.index] = ratio.values

    return result


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume Weighted Average Price with daily reset.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3

    For intraday data, resets each day.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        Series of VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if hasattr(df.index, "date"):
        dates = pd.Series(df.index.date, index=df.index)
        unique_dates = dates.nunique()
        if unique_dates < len(df) * 0.9:
            cum_tp_vol = tp_volume.groupby(dates).cumsum()
            cum_vol = df["volume"].groupby(dates).cumsum()
        else:
            cum_tp_vol = tp_volume.cumsum()
            cum_vol = df["volume"].cumsum()
    else:
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = df["volume"].cumsum()

    cum_vol = cum_vol.replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol
    return vwap.fillna(typical_price)


def compute_support_resistance(
    df: pd.DataFrame, pivot_window: int = 20, num_levels: int = 5
) -> pd.DataFrame:
    """Compute support and resistance levels.

    Uses rolling pivot points and volume-weighted price levels.

    Args:
        df: DataFrame with OHLCV data.
        pivot_window: Window for detecting local highs/lows.
        num_levels: Number of S/R levels to track.

    Returns:
        DataFrame with added support/resistance columns.
    """
    df = df.copy()

    # Rolling pivot-based support/resistance
    df["resistance_1"] = df["high"].rolling(window=pivot_window, min_periods=1).max()
    df["support_1"] = df["low"].rolling(window=pivot_window, min_periods=1).min()

    # Volume-weighted mean as dynamic level
    window_vol = df["volume"].rolling(window=pivot_window, min_periods=1).sum()
    window_vol = window_vol.replace(0, np.nan)
    weighted_price = (
        (df["close"] * df["volume"]).rolling(window=pivot_window, min_periods=1).sum()
        / window_vol
    )
    df["resistance_2"] = weighted_price + (df["resistance_1"] - weighted_price) * 0.5
    df["support_2"] = weighted_price - (weighted_price - df["support_1"]) * 0.5

    # Nearest S/R distance (normalized by price)
    price = df["close"]
    dist_to_support = (price - df["support_1"]).abs()
    dist_to_resistance = (df["resistance_1"] - price).abs()
    nearest_dist = pd.concat([dist_to_support, dist_to_resistance], axis=1).min(axis=1)
    df["nearest_sr_distance"] = nearest_dist / price

    return df


def compute_delta_zscore(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Compute rolling Z-score of delta for dynamic exit management.

    Z-score = (current_delta - rolling_mean_delta) / rolling_std_delta

    Measures how extreme the current order flow signal is relative to
    recent history. Used to determine extended take profit levels.

    Args:
        df: DataFrame with 'volume_delta' column.
        window: Rolling window size (default 50 bars).

    Returns:
        Series of Z-score values. NaN-filled for insufficient data.
    """
    delta = df["volume_delta"]
    rolling_mean = delta.rolling(window=window, min_periods=1).mean()
    rolling_std = delta.rolling(window=window, min_periods=1).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    zscore = (delta - rolling_mean) / rolling_std
    return zscore.fillna(0.0)


def compute_absorption_signal(df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Detect REAL absorption: bars where BOTH bid and ask volume are elevated.

    True absorption means both buyers and sellers are fighting heavily
    at the same price level. The absorbing side (the one that holds)
    wins the battle.

    A bar is absorption when:
    - bid_volume > threshold * rolling_avg(bid_volume)
    - ask_volume > threshold * rolling_avg(ask_volume)
    - BOTH conditions simultaneously

    Direction:
    - At support (price near support_1): bullish absorption (+1)
      Buyers absorbing selling pressure
    - At resistance (price near resistance_1): bearish absorption (-1)
      Sellers absorbing buying pressure

    Args:
        df: DataFrame with bid_volume, ask_volume, support_1, resistance_1.
        threshold: Multiplier for rolling average (default 2.0 = both sides
                   must have 2x their average volume).

    Returns:
        Series: +1 (bullish absorption at support),
                -1 (bearish absorption at resistance), 0 (none).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if "bid_volume" not in df.columns or "ask_volume" not in df.columns:
        return result

    bid_vol = df["bid_volume"].astype(float)
    ask_vol = df["ask_volume"].astype(float)

    # Rolling averages
    bid_avg = bid_vol.rolling(window=20, min_periods=1).mean()
    ask_avg = ask_vol.rolling(window=20, min_periods=1).mean()

    # Both sides elevated simultaneously
    bid_elevated = bid_vol > (bid_avg * threshold)
    ask_elevated = ask_vol > (ask_avg * threshold)
    absorption = bid_elevated & ask_elevated

    # Determine direction based on proximity to S/R
    if "support_1" in df.columns and "resistance_1" in df.columns:
        dist_to_support = (df["close"] - df["support_1"]).abs()
        dist_to_resistance = (df["resistance_1"] - df["close"]).abs()
        near_support = dist_to_support < dist_to_resistance

        result[absorption & near_support] = 1   # Bullish: absorbing at support
        result[absorption & ~near_support] = -1  # Bearish: absorbing at resistance
    else:
        # Without S/R context, use delta direction
        result[absorption & (df["delta"] > 0)] = 1
        result[absorption & (df["delta"] < 0)] = -1

    return result


def compute_imbalance_signal(
    df: pd.DataFrame, ratio: float = 3.0
) -> pd.Series:
    """Detect REAL imbalance: bars where one side dominates heavily.

    An imbalance occurs when the bid/ask ratio exceeds the threshold
    in one direction.

    - bid_volume / ask_volume > ratio: buying imbalance (+1)
    - ask_volume / bid_volume > ratio: selling imbalance (-1)

    Args:
        df: DataFrame with bid_volume, ask_volume columns.
        ratio: Minimum ratio threshold (default 3.0 = one side 3x the other).

    Returns:
        Series: +1 (buying imbalance), -1 (selling imbalance), 0 (balanced).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if "bid_volume" not in df.columns or "ask_volume" not in df.columns:
        return result

    bid_vol = df["bid_volume"].astype(float)
    ask_vol = df["ask_volume"].astype(float)

    # Avoid division by zero
    safe_ask = ask_vol.replace(0, 1)
    safe_bid = bid_vol.replace(0, 1)

    buy_ratio = bid_vol / safe_ask
    sell_ratio = ask_vol / safe_bid

    result[buy_ratio >= ratio] = 1   # Buying imbalance
    result[sell_ratio >= ratio] = -1  # Selling imbalance

    return result
