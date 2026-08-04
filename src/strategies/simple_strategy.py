"""Simple strategy: heavy-volume level + real order flow confirmation.

This is the core idea shared by both source methodologies, stripped of
everything else:

1. LEVEL (volume profile): price is at/near the heavy-volume node -- the
   price zone where the most volume traded over a lookback window. That is
   where institutions positioned.
2. CONFIRMATION (real order flow): the real delta at that level says which
   side is winning. Positive delta at a support level = long, negative delta
   at a resistance level = short.

Exits are fixed points only: a fixed stop and a fixed target. No partial
closes, no trailing, no staged advancement, no time-based exit.

Everything here is strictly causal: the value for bar i is computed from
bars <= i only.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.strategies.base import BaseStrategy

# Width of a volume-profile price bin, in index points. Fixed at 5 points
# because NQ round-number levels cluster on 5-point increments and 5 points
# is finer than the smallest level_proximity_points we ever evaluate.
# Deliberately NOT part of the tunable parameter surface.
PROFILE_BIN_WIDTH_POINTS = 5.0

# Rolling window (bars) for the delta Z-score. 50 bars is ~4 hours on 5-min
# data (a bit under one session), the same window the preprocessor uses.
# Deliberately NOT part of the tunable parameter surface.
DELTA_ZSCORE_WINDOW = 50


def heavy_volume_node(
    df: pd.DataFrame,
    lookback: int,
    bin_width: float = PROFILE_BIN_WIDTH_POINTS,
) -> pd.Series:
    """Compute the heavy-volume node price for every bar.

    For bar i, volume traded over bars [i - lookback + 1, i] is bucketed into
    fixed-width price bins by each bar's typical price ((H + L + C) / 3). The
    returned level is the centre of the bin holding the most volume.

    Strictly causal: the value at bar i uses bars <= i only, and is unchanged
    if every bar after i is removed.

    Args:
        df: DataFrame with high, low, close, volume columns.
        lookback: Number of bars in the volume profile window.
        bin_width: Price-bin width in index points.

    Returns:
        Series of heavy-volume node prices (NaN during the warmup window).
    """
    n = len(df)
    node = np.full(n, np.nan)
    lookback = int(lookback)

    if lookback < 1 or n < lookback:
        return pd.Series(node, index=df.index)

    typical = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    bin_idx = np.floor(typical / bin_width).astype(np.int64)

    windowed_bins = sliding_window_view(bin_idx, lookback)
    windowed_vol = sliding_window_view(volume, lookback)

    # Bins are made relative to each window's own lowest bin so the histogram
    # width stays small and each window is independent of the others.
    base = windowed_bins.min(axis=1)
    relative = windowed_bins - base[:, None]
    width = int(relative.max()) + 1

    n_windows = relative.shape[0]
    # Cap the working histogram at ~4M cells per chunk to bound memory.
    chunk_rows = max(1, int(4_000_000 // width))

    for start in range(0, n_windows, chunk_rows):
        stop = min(start + chunk_rows, n_windows)
        rows = stop - start
        offsets = np.arange(rows, dtype=np.int64)[:, None] * width
        flat = (offsets + relative[start:stop]).ravel()
        totals = np.bincount(
            flat,
            weights=windowed_vol[start:stop].ravel(),
            minlength=rows * width,
        ).reshape(rows, width)
        best_bin = totals.argmax(axis=1)
        node[lookback - 1 + start : lookback - 1 + stop] = (
            base[start:stop] + best_bin + 0.5
        ) * bin_width

    return pd.Series(node, index=df.index)


def rolling_delta_zscore(
    df: pd.DataFrame, window: int = DELTA_ZSCORE_WINDOW
) -> pd.Series:
    """Compute the rolling Z-score of real order flow delta.

    Uses the real delta column (bid_volume - ask_volume) when present, so a
    positive Z-score means net buying. Strictly causal: bar i uses the
    trailing window ending at bar i.

    Args:
        df: DataFrame with a 'delta' (preferred) or 'volume_delta' column.
        window: Rolling window length in bars.

    Returns:
        Series of delta Z-scores (NaN during the warmup window).
    """
    if "delta" in df.columns:
        delta = df["delta"].astype(float)
    elif "volume_delta" in df.columns:
        delta = df["volume_delta"].astype(float)
    else:
        raise KeyError("DataFrame needs a 'delta' or 'volume_delta' column")

    mean = delta.rolling(window=window, min_periods=window).mean()
    std = delta.rolling(window=window, min_periods=window).std(ddof=0)

    return (delta - mean) / std.replace(0.0, np.nan)


class SimpleStrategy(BaseStrategy):
    """Heavy-volume level + delta confirmation, fixed-point exits.

    Two entry conditions, nothing else:
    - price is within level_proximity_points of the heavy-volume node
    - the delta Z-score confirms the side the level is being defended from

    The node acts as support when price sits at or above it and as resistance
    when price sits at or below it, so a long needs buying pressure at
    support and a short needs selling pressure at resistance.
    """

    def default_params(self) -> dict[str, Any]:
        """Return default parameters (the entire tunable surface)."""
        return {
            # Bars used to build the volume profile (78 = one 6.5h session).
            "profile_lookback": 78,
            # How close price must be to the node to count as "at the level".
            "level_proximity_points": 10,
            # Minimum |delta Z-score| for order flow confirmation.
            "delta_threshold": 1.0,
            # Fixed stop distance in points.
            "stop_points": 20,
            # Fixed target distance in points.
            "target_points": 30,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry signals from level proximity plus delta confirmation.

        Args:
            df: Preprocessed DataFrame with OHLCV and delta columns.

        Returns:
            DataFrame with 'signal' (1 long, -1 short, 0 flat), 'level' and
            'delta_zscore' columns.
        """
        lookback = int(self.params["profile_lookback"])
        proximity = float(self.params["level_proximity_points"])
        threshold = float(self.params["delta_threshold"])

        level = heavy_volume_node(df, lookback)
        zscore = rolling_delta_zscore(df)

        # Distance from the level: positive means price is above the node.
        distance = df["close"] - level
        at_level = distance.abs() <= proximity

        long_entry = at_level & (distance >= 0) & (zscore >= threshold)
        short_entry = at_level & (distance <= 0) & (zscore <= -threshold)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[long_entry] = 1
        signal[short_entry] = -1

        result = pd.DataFrame(index=df.index)
        result["signal"] = signal
        result["level"] = level
        result["delta_zscore"] = zscore
        return result

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Return the fixed stop price for an entry at bar idx."""
        entry_price = float(df["close"].iloc[idx])
        return entry_price - direction * float(self.params["stop_points"])

    def get_take_profit(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: int,
        feature_zscore: float | None = None,
    ) -> float:
        """Return the fixed target price for an entry at bar idx.

        feature_zscore is accepted for interface compatibility and ignored:
        the target is always a fixed number of points.
        """
        entry_price = float(df["close"].iloc[idx])
        return entry_price + direction * float(self.params["target_points"])

    def get_param_ranges(self) -> dict[str, list]:
        """Return the signal parameter grid used by walk-forward windows.

        Stop and target are locked by the train/validation selection, so only
        the two signal parameters are re-optimized inside each walk-forward
        window.
        """
        return {
            "delta_threshold": [0.5, 1.0, 1.5],
            "level_proximity_points": [5, 10],
        }
