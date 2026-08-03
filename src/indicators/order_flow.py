"""Order flow proxy indicators.

Implements order flow concepts using OHLCV data as proxies for
true order flow (bid/ask, Level 2) which is not freely available.
"""

import numpy as np
import pandas as pd


def detect_absorption(
    df: pd.DataFrame, threshold: float = 2.0
) -> pd.Series:
    """Detect absorption bars where heavy volume appears at S/R zones.

    Absorption occurs when unusually heavy volume appears at key
    support/resistance zones, signaling that buying/selling pressure
    is being absorbed and a reversal is likely.

    Args:
        df: DataFrame with columns: relative_volume, nearest_sr_distance,
            close, open, support_1, resistance_1.
        threshold: Minimum relative volume multiplier for detection.

    Returns:
        Series with values: 1 (bullish absorption at support),
        -1 (bearish absorption at resistance), 0 (no absorption).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    # High volume condition
    high_volume = df["relative_volume"] >= threshold

    # Near support/resistance (within a fraction of price)
    sr_proximity = df["nearest_sr_distance"] < 0.03  # Within 3% of price

    # Determine if near support or resistance
    dist_to_support = (df["close"] - df["support_1"]).abs()
    dist_to_resistance = (df["resistance_1"] - df["close"]).abs()

    near_support = dist_to_support < dist_to_resistance
    near_resistance = ~near_support

    # Bullish absorption: high volume at support with bullish close
    bullish = high_volume & sr_proximity & near_support & (df["close"] > df["open"])
    result[bullish] = 1

    # Bearish absorption: high volume at resistance with bearish close
    bearish = high_volume & sr_proximity & near_resistance & (df["close"] < df["open"])
    result[bearish] = -1

    return result


def cumulative_delta_divergence(
    df: pd.DataFrame, lookback: int = 10
) -> pd.Series:
    """Detect cumulative delta divergence from price.

    When price makes new highs but cumulative delta is declining
    (bearish divergence) or price makes new lows but cumulative delta
    is rising (bullish divergence).

    Args:
        df: DataFrame with columns: close, cumulative_delta.
        lookback: Number of bars to look back for divergence detection.

    Returns:
        Series with values: 1 (bullish divergence - price low, delta rising),
        -1 (bearish divergence - price high, delta falling), 0 (no divergence).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if len(df) < lookback + 1:
        return result

    close = df["close"]
    cum_delta = df["cumulative_delta"]

    # Price making new highs over lookback window
    price_high = close == close.rolling(window=lookback, min_periods=lookback).max()
    # Cumulative delta declining (current < lookback bars ago)
    delta_declining = cum_delta < cum_delta.shift(lookback)

    # Price making new lows over lookback window
    price_low = close == close.rolling(window=lookback, min_periods=lookback).min()
    # Cumulative delta rising
    delta_rising = cum_delta > cum_delta.shift(lookback)

    # Bearish divergence: price at highs, delta declining
    bearish_div = price_high & delta_declining
    result[bearish_div] = -1

    # Bullish divergence: price at lows, delta rising
    bullish_div = price_low & delta_rising
    result[bullish_div] = 1

    return result


def detect_stacked_imbalances(
    df: pd.DataFrame, ratio: float = 2.0, min_bars: int = 2
) -> pd.Series:
    """Detect stacked imbalances where one side dominates consecutively.

    When volume delta is heavily skewed in the same direction for
    multiple consecutive bars (stacked), this creates strong
    support/resistance zones.

    Args:
        df: DataFrame with columns: volume_delta, volume.
        ratio: Minimum ratio of absolute delta to average for imbalance.
        min_bars: Minimum consecutive bars needed for a stacked signal.

    Returns:
        Series with values: 1 (bullish stacked imbalances),
        -1 (bearish stacked imbalances), 0 (no signal).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if len(df) < min_bars:
        return result

    # Compute average absolute delta for reference
    avg_abs_delta = df["volume_delta"].abs().rolling(window=20, min_periods=1).mean()
    avg_abs_delta = avg_abs_delta.replace(0, np.nan).fillna(1)

    # Check if current bar has significant positive/negative imbalance
    bullish_imbalance = df["volume_delta"] > (avg_abs_delta * ratio)
    bearish_imbalance = df["volume_delta"] < -(avg_abs_delta * ratio)

    # Count consecutive bullish/bearish imbalances
    bullish_count = pd.Series(0, index=df.index, dtype=int)
    bearish_count = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df)):
        if bullish_imbalance.iloc[i]:
            bullish_count.iloc[i] = (bullish_count.iloc[i - 1] + 1) if i > 0 else 1
        if bearish_imbalance.iloc[i]:
            bearish_count.iloc[i] = (bearish_count.iloc[i - 1] + 1) if i > 0 else 1

    # Signal when consecutive count reaches min_bars
    result[bullish_count >= min_bars] = 1
    result[bearish_count >= min_bars] = -1

    return result


def detect_failed_auctions(
    df: pd.DataFrame, threshold: float = 0.1
) -> pd.Series:
    """Detect failed auctions where price levels get revisited.

    A failed auction occurs when a candle's extreme (high or low) shows
    volume anomaly suggesting the auction didn't complete properly.
    These levels tend to be revisited (act as price magnets).

    Args:
        df: DataFrame with columns: high, low, close, volume, relative_volume.
        threshold: Minimum price proximity for detection (fraction of range).

    Returns:
        Series with values: 1 (failed auction below - bullish target),
        -1 (failed auction above - bearish target), 0 (no signal).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if len(df) < 3:
        return result

    # Detect bars where closing was very near the high or low
    # with relatively low volume (incomplete auction)
    bar_range = df["high"] - df["low"]
    bar_range = bar_range.replace(0, np.nan).fillna(1)

    # Close near high but next bar gaps down (failed upside auction)
    close_near_high = (df["high"] - df["close"]) / bar_range < threshold
    next_bearish = df["close"].shift(-1) < df["close"]
    low_vol_extreme = df["relative_volume"] < 1.0

    # Failed auction at high: close near high, low volume, next bar reverses
    # The signal uses bar i+1's close (via shift(-1)), so we must shift
    # the result forward by 1 bar to avoid lookahead bias. The signal
    # only becomes available after the confirming bar closes.
    failed_high = close_near_high & low_vol_extreme & next_bearish
    failed_high_signal = failed_high.shift(1).fillna(False)
    result[failed_high_signal] = -1

    # Close near low but next bar gaps up (failed downside auction)
    close_near_low = (df["close"] - df["low"]) / bar_range < threshold
    next_bullish = df["close"].shift(-1) > df["close"]

    # Failed auction at low: close near low, low volume, next bar reverses
    # Same forward-shift to eliminate lookahead (consistent with trapped traders)
    failed_low = close_near_low & low_vol_extreme & next_bullish
    failed_low_signal = failed_low.shift(1).fillna(False)
    result[failed_low_signal] = 1

    return result


def detect_trapped_traders(
    df: pd.DataFrame, lookback: int = 5
) -> pd.Series:
    """Detect trapped traders based on volume at extremes followed by reversal.

    When heavy buying volume appears at the top of a candle but the next
    candle is bearish, those buyers are trapped. Go short.
    Vice versa for trapped sellers at the bottom.

    Args:
        df: DataFrame with columns: open, high, low, close, volume,
            volume_delta, relative_volume.
        lookback: Number of bars for volume reference.

    Returns:
        Series with values: 1 (trapped sellers - go long),
        -1 (trapped buyers - go short), 0 (no signal).
    """
    result = pd.Series(0, index=df.index, dtype=int)

    if len(df) < 2:
        return result

    # High volume bullish bar (strong buying)
    strong_buying = (
        (df["volume_delta"] > 0)
        & (df["close"] > df["open"])
        & (df["relative_volume"] > 1.5)
    )

    # High volume bearish bar (strong selling)
    strong_selling = (
        (df["volume_delta"] < 0)
        & (df["close"] < df["open"])
        & (df["relative_volume"] > 1.5)
    )

    # Next bar reverses
    next_bearish = df["close"].shift(-1) < df["open"].shift(-1)
    next_bullish = df["close"].shift(-1) > df["open"].shift(-1)

    # Trapped buyers: strong buying followed by bearish reversal
    # Signal appears on the NEXT bar (the reversal bar)
    trapped_buyers = strong_buying & next_bearish
    # Shift forward to signal on the reversal bar
    trapped_buyers_signal = trapped_buyers.shift(1).fillna(False)
    result[trapped_buyers_signal] = -1

    # Trapped sellers: strong selling followed by bullish reversal
    trapped_sellers = strong_selling & next_bullish
    trapped_sellers_signal = trapped_sellers.shift(1).fillna(False)
    result[trapped_sellers_signal] = 1

    return result
