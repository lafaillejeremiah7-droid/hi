"""Volume profile indicators.

Implements volume-at-price analysis to identify high-volume nodes,
accumulation zones, and support/resistance levels based on
where the most trading activity occurred.
"""

import numpy as np
import pandas as pd


def compute_volume_profile(
    df: pd.DataFrame, window: int = 100, bins: int = 50
) -> pd.DataFrame:
    """Build volume-at-price distribution over rolling windows.

    Returns Point of Control (POC), Value Area High (VAH), and
    Value Area Low (VAL) for each bar based on the preceding window.

    Optimized: computes profile every 12 bars (1 hour on 5-min data)
    and forward-fills intermediate bars to reduce computation by ~12x.

    Args:
        df: DataFrame with columns: high, low, close, volume.
        window: Lookback period for volume profile calculation.
        bins: Number of price bins for the histogram.

    Returns:
        DataFrame with columns: poc (Point of Control), vah (Value Area High),
        val (Value Area Low).
    """
    poc = pd.Series(np.nan, index=df.index)
    vah = pd.Series(np.nan, index=df.index)
    val = pd.Series(np.nan, index=df.index)

    # Compute every step_size bars for performance (reuse profile within step)
    step_size = 12  # Recompute every hour on 5-min data

    # Pre-extract arrays for speed
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    for i in range(window, len(df), step_size):
        seg_start = i - window
        seg_highs = highs[seg_start:i]
        seg_lows = lows[seg_start:i]
        seg_closes = closes[seg_start:i]
        seg_volumes = volumes[seg_start:i]

        price_low = seg_lows.min()
        price_high = seg_highs.max()

        if price_high == price_low:
            poc.iloc[i] = price_low
            vah.iloc[i] = price_high
            val.iloc[i] = price_low
            continue

        # Create price bins
        bin_edges = np.linspace(price_low, price_high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Vectorized volume distribution using typical price
        typical = (seg_highs + seg_lows + seg_closes) / 3
        bin_indices = ((typical - price_low) / (price_high - price_low) * (bins - 1)).astype(int)
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        vol_profile = np.zeros(bins)
        np.add.at(vol_profile, bin_indices, seg_volumes)

        # POC: price level with highest volume
        poc_idx = np.argmax(vol_profile)
        poc.iloc[i] = bin_centers[poc_idx]

        # Value Area: 70% of volume around POC
        total_vol = vol_profile.sum()
        if total_vol == 0:
            vah.iloc[i] = price_high
            val.iloc[i] = price_low
            continue

        va_vol = vol_profile[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while va_vol < total_vol * 0.7:
            expand_up = (
                vol_profile[va_high_idx + 1] if va_high_idx + 1 < bins else 0
            )
            expand_down = vol_profile[va_low_idx - 1] if va_low_idx - 1 >= 0 else 0

            if expand_up >= expand_down and va_high_idx + 1 < bins:
                va_high_idx += 1
                va_vol += vol_profile[va_high_idx]
            elif va_low_idx - 1 >= 0:
                va_low_idx -= 1
                va_vol += vol_profile[va_low_idx]
            else:
                break

        vah.iloc[i] = bin_centers[va_high_idx]
        val.iloc[i] = bin_centers[va_low_idx]

    # Forward-fill to cover bars between computation steps
    poc = poc.ffill()
    vah = vah.ffill()
    val = val.ffill()

    result = pd.DataFrame({"poc": poc, "vah": vah, "val": val}, index=df.index)
    return result


def find_high_volume_nodes(
    df: pd.DataFrame, window: int = 100, threshold: float = 0.7, bins: int = 50
) -> pd.Series:
    """Identify price levels with significantly above-average volume.

    High Volume Nodes (HVN) represent price levels where significant
    trading occurred, acting as support/resistance.

    Optimized: computes every 12 bars and forward-fills for performance.

    Args:
        df: DataFrame with OHLCV columns.
        window: Lookback period.
        threshold: Minimum fraction of max bin volume to qualify as HVN.
        bins: Number of price bins.

    Returns:
        Series of the nearest HVN price level for each bar.
    """
    hvn = pd.Series(np.nan, index=df.index)
    step_size = 12  # Recompute every hour on 5-min data

    # Pre-extract arrays for speed
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    for i in range(window, len(df), step_size):
        seg_start = i - window
        seg_highs = highs[seg_start:i]
        seg_lows = lows[seg_start:i]
        seg_closes = closes[seg_start:i]
        seg_volumes = volumes[seg_start:i]

        price_low = seg_lows.min()
        price_high = seg_highs.max()

        if price_high == price_low:
            hvn.iloc[i] = price_low
            continue

        bin_edges = np.linspace(price_low, price_high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Vectorized volume distribution
        typical = (seg_highs + seg_lows + seg_closes) / 3
        bin_indices = ((typical - price_low) / (price_high - price_low) * (bins - 1)).astype(int)
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        vol_profile = np.zeros(bins)
        np.add.at(vol_profile, bin_indices, seg_volumes)

        max_vol = vol_profile.max()
        if max_vol == 0:
            hvn.iloc[i] = closes[i]
            continue

        # Find HVN levels (bins above threshold * max)
        hvn_mask = vol_profile >= (threshold * max_vol)
        hvn_levels = bin_centers[hvn_mask]

        if len(hvn_levels) > 0:
            # Return the nearest HVN to current price
            current_price = closes[i]
            distances = np.abs(hvn_levels - current_price)
            hvn.iloc[i] = hvn_levels[np.argmin(distances)]
        else:
            hvn.iloc[i] = bin_centers[np.argmax(vol_profile)]

    # Forward-fill to cover bars between steps
    hvn = hvn.ffill()

    return hvn


def detect_volume_accumulation_zones(
    df: pd.DataFrame,
    min_bars: int = 10,
    max_range_atr: float = 2.0,
    atr_period: int = 14,
) -> pd.Series:
    """Find consolidation/rotation zones where heavy volume accumulated.

    A rotation zone is characterized by:
    - Price moving in a narrow range (< max_range_atr * ATR)
    - Above-average volume (institutions accumulating)
    - Followed by a breakout

    Optimized with vectorized rolling window operations for large datasets.

    Args:
        df: DataFrame with OHLCV data.
        min_bars: Minimum bars for a rotation zone.
        max_range_atr: Maximum range as multiple of ATR.
        atr_period: Period for ATR calculation.

    Returns:
        Series with values: 1 (accumulation zone detected, bullish breakout expected),
        -1 (distribution zone, bearish breakout expected), 0 (no signal).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if len(df) < min_bars + atr_period:
        return result

    # Compute ATR vectorized
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # Compute rolling range over min_bars window
    rolling_high = df["high"].rolling(window=min_bars, min_periods=min_bars).max()
    rolling_low = df["low"].rolling(window=min_bars, min_periods=min_bars).min()
    price_range = rolling_high - rolling_low

    # Compute rolling average volume over min_bars
    rolling_vol = df["volume"].rolling(window=min_bars, min_periods=min_bars).mean()

    # Expanding average volume (overall average up to current bar)
    expanding_avg_vol = df["volume"].expanding(min_periods=1).mean()

    # Rotation condition: tight range relative to ATR
    rotation_mask = (
        (price_range < max_range_atr * atr)
        & (atr > 0)
        & (rolling_vol > expanding_avg_vol)
    )

    # Only valid after warmup period
    warmup_idx = min_bars + atr_period
    rotation_mask.iloc[:warmup_idx] = False

    # Breakout detection: price breaks above/below the rolling range
    bullish_breakout = rotation_mask & (df["close"] > rolling_high.shift(1))
    bearish_breakout = rotation_mask & (df["close"] < rolling_low.shift(1))

    result[bullish_breakout] = 1
    result[bearish_breakout] = -1

    return result


def identify_trend_volume_clusters(
    df: pd.DataFrame,
    trend_period: int = 20,
    min_trend_strength: float = 25.0,
    atr_period: int = 14,
) -> pd.Series:
    """During trends, find volume clusters where institutions added positions.

    Identifies significant volume bumps within a trend that represent
    institutional position building. These levels act as support in
    pullbacks.

    Optimized: computes every 6 bars and uses vectorized numpy operations.

    Args:
        df: DataFrame with OHLCV data.
        trend_period: Period for trend detection.
        min_trend_strength: Minimum trend strength (price change / ATR).
        atr_period: Period for ATR calculation.

    Returns:
        Series of cluster price levels (NaN where no cluster detected).
    """
    cluster_levels = pd.Series(np.nan, index=df.index)

    if len(df) < trend_period + atr_period:
        return cluster_levels

    # Compute ATR vectorized
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    # Compute trend strength using price change relative to ATR
    price_change = df["close"] - df["close"].shift(trend_period)
    trend_strength = price_change.abs() / atr.replace(0, np.nan)

    # Normalized threshold
    norm_threshold = min_trend_strength / 10

    # Pre-extract arrays for speed
    closes = df["close"].values
    volumes = df["volume"].values
    trend_strength_vals = trend_strength.values

    step_size = 6  # Compute every 6 bars for performance

    start_idx = trend_period + atr_period
    for i in range(start_idx, len(df), step_size):
        # Check if we're in a trend
        ts = trend_strength_vals[i]
        if np.isnan(ts) or ts < norm_threshold:
            continue

        # Look at volume in the trend segment
        seg_start = i - trend_period
        seg_volumes = volumes[seg_start:i]
        seg_closes = closes[seg_start:i]
        avg_vol = seg_volumes.mean()

        # Find bars with volume significantly above average (cluster)
        high_vol_mask = seg_volumes > avg_vol * 1.5
        if high_vol_mask.any():
            high_vol_vols = seg_volumes[high_vol_mask]
            high_vol_closes = seg_closes[high_vol_mask]
            # Volume-weighted average price of the cluster
            cluster_price = (high_vol_closes * high_vol_vols).sum() / high_vol_vols.sum()
            cluster_levels.iloc[i] = cluster_price

    return cluster_levels


def support_resistance_flip(
    df: pd.DataFrame, confirmation_bars: int = 3
) -> pd.Series:
    """Detect when support zones become resistance and vice versa.

    When a heavy volume support zone gets breached, it becomes resistance.
    When a heavy volume resistance zone gets breached, it becomes support.

    Optimized with vectorized rolling operations.

    Args:
        df: DataFrame with columns: close, support_1, resistance_1.
        confirmation_bars: Number of bars below/above for confirmation.

    Returns:
        Series with values: 1 (resistance flipped to support - bullish),
        -1 (support flipped to resistance - bearish), 0 (no flip).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if len(df) < confirmation_bars + 1:
        return result

    support = df["support_1"]
    resistance = df["resistance_1"]
    close = df["close"]

    # Vectorized: check if price has been below shifted support for N bars
    prev_support = support.shift(confirmation_bars)
    prev_resistance = resistance.shift(confirmation_bars)

    # For bearish flip: all bars in window below previous support
    below_support = close < prev_support
    # Rolling min over confirmation_bars: if all True (all below), rolling min == 1
    bars_below = below_support.rolling(window=confirmation_bars, min_periods=confirmation_bars).min()
    # And current bar is bouncing (close > prev close)
    bouncing_up = close > close.shift(1)
    bearish_flip = (bars_below == 1) & bouncing_up

    # For bullish flip: all bars in window above previous resistance
    above_resistance = close > prev_resistance
    bars_above = above_resistance.rolling(window=confirmation_bars, min_periods=confirmation_bars).min()
    # And current bar is pulling back (close < prev close)
    pulling_back = close < close.shift(1)
    bullish_flip = (bars_above == 1) & pulling_back

    result[bearish_flip] = -1
    result[bullish_flip] = 1

    return result
