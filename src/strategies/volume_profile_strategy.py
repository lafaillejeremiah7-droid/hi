"""Volume Profile Strategy implementation.

Implements three setups based on volume profile analysis:
1. Volume Accumulation Setup (rotation zone breakout + pullback)
2. Trend Setup (trend + volume cluster pullback)
3. Support/Resistance Flip
"""

from typing import Any

import numpy as np
import pandas as pd

from src.indicators.volume_profile import (
    compute_volume_profile,
    detect_volume_accumulation_zones,
    find_high_volume_nodes,
    identify_trend_volume_clusters,
    support_resistance_flip,
)
from src.strategies.base import BaseStrategy


class VolumeProfileStrategy(BaseStrategy):
    """Volume profile-based trading strategy.

    Generates signals from three distinct setups:
    1. Accumulation: Heavy volume rotation followed by breakout,
       enter on pullback to rotation's high-volume node.
    2. Trend: Strong trend with volume clusters, enter on pullback
       to cluster peak.
    3. S/R Flip: Support becomes resistance or vice versa,
       combined with volume cluster confirmation.
    """

    def default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "profile_bins": 50,
            "profile_lookback": 100,
            "min_volume_concentration": 0.7,
            "trend_period": 20,
            "min_trend_strength": 25.0,
            "rotation_max_range_atr": 2.0,
            "rotation_min_bars": 10,
            "sr_flip_confirmation_bars": 3,
            "stop_loss_atr_mult": 1.0,
            "take_profit_lookback": 50,
            "atr_period": 14,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from volume profile setups.

        Combines accumulation, trend, and S/R flip setups.

        Args:
            df: Preprocessed DataFrame with OHLCV and indicator columns.

        Returns:
            DataFrame with 'signal' and 'setup_type' columns.
        """
        params = self.params
        result = pd.DataFrame(index=df.index)
        signal = pd.Series(0, index=df.index, dtype=int)
        setup_type = pd.Series("", index=df.index, dtype=str)

        # Setup 1: Volume Accumulation
        accum_signals = detect_volume_accumulation_zones(
            df,
            min_bars=params["rotation_min_bars"],
            max_range_atr=params["rotation_max_range_atr"],
            atr_period=params["atr_period"],
        )

        # Setup 2: Trend Volume Clusters
        cluster_levels = identify_trend_volume_clusters(
            df,
            trend_period=params["trend_period"],
            min_trend_strength=params["min_trend_strength"],
            atr_period=params["atr_period"],
        )

        # Detect pullback to cluster level
        trend_signals = self._detect_cluster_pullback(df, cluster_levels)

        # Setup 3: S/R Flip
        if "support_1" in df.columns and "resistance_1" in df.columns:
            sr_flip_signals = support_resistance_flip(
                df, confirmation_bars=params["sr_flip_confirmation_bars"]
            )
        else:
            sr_flip_signals = pd.Series(0, index=df.index, dtype=int)

        # Combine signals with priority: accumulation > trend > S/R flip
        # Accumulation signals
        accum_long = accum_signals == 1
        accum_short = accum_signals == -1
        signal[accum_long] = 1
        signal[accum_short] = -1
        setup_type[accum_long] = "accumulation"
        setup_type[accum_short] = "accumulation"

        # Trend signals (don't override accumulation)
        no_signal = signal == 0
        trend_long = (trend_signals == 1) & no_signal
        trend_short = (trend_signals == -1) & no_signal
        signal[trend_long] = 1
        signal[trend_short] = -1
        setup_type[trend_long] = "trend"
        setup_type[trend_short] = "trend"

        # S/R flip signals (lowest priority)
        no_signal = signal == 0
        sr_long = (sr_flip_signals == 1) & no_signal
        sr_short = (sr_flip_signals == -1) & no_signal
        signal[sr_long] = 1
        signal[sr_short] = -1
        setup_type[sr_long] = "sr_flip"
        setup_type[sr_short] = "sr_flip"

        result["signal"] = signal
        result["setup_type"] = setup_type

        return result

    def _detect_cluster_pullback(
        self, df: pd.DataFrame, cluster_levels: pd.Series
    ) -> pd.Series:
        """Detect pullbacks to volume cluster levels (vectorized).

        Uses forward-fill of cluster levels and vectorized distance
        checks instead of per-bar loops for performance on large datasets.

        Args:
            df: DataFrame with OHLCV data.
            cluster_levels: Series of cluster price levels.

        Returns:
            Series with 1 (bullish pullback to cluster), -1 (bearish), 0.
        """
        result = pd.Series(0, index=df.index, dtype=int)

        if cluster_levels.isna().all():
            return result

        # Compute ATR vectorized
        atr_period = self.params["atr_period"]
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=atr_period, min_periods=1).mean()

        # Determine trend direction vectorized
        trend_period = self.params["trend_period"]
        trend_dir = np.sign(df["close"] - df["close"].shift(trend_period))

        # Forward-fill cluster levels to get "most recent cluster" at each bar
        # Use limit=trend_period to only fill within the lookback window
        recent_cluster = cluster_levels.ffill(limit=trend_period)

        # Vectorized distance check: price near cluster level within 1 ATR
        distance = (df["close"] - recent_cluster).abs()
        near_cluster = (distance < atr) & atr.gt(0) & recent_cluster.notna()

        # Apply trend direction filter (only signal after trend_period warmup)
        valid = near_cluster & (df.index >= df.index[min(trend_period, len(df) - 1)])

        # Assign signals based on trend direction
        long_mask = valid & (trend_dir > 0)
        short_mask = valid & (trend_dir < 0)
        result[long_mask] = 1
        result[short_mask] = -1

        return result

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Calculate stop loss price.

        Supports two modes:
        - "fixed": entry +/- fixed_points (from exit_management config)
        - "atr": entry +/- ATR * multiplier (legacy, places stop in low-volume area)

        Args:
            df: Full DataFrame.
            idx: Entry bar index position.
            direction: 1 for long, -1 for short.

        Returns:
            Stop loss price.
        """
        entry_price = df["close"].iloc[idx]
        mode = self.params.get("stop_loss_mode", "atr")

        if mode == "fixed":
            fixed_points = self.params.get("stop_loss_fixed_points", 20)
            if direction == 1:
                return entry_price - fixed_points
            else:
                return entry_price + fixed_points
        else:
            atr = self._compute_atr(df, idx)
            mult = self.params["stop_loss_atr_mult"]
            if direction == 1:
                return entry_price - atr * mult
            else:
                return entry_price + atr * mult

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int, feature_zscore: float | None = None) -> float:
        """Calculate take profit price.

        Supports two modes:
        - "fixed": entry +/- fixed_points (30 default, 50 when high Z-score)
        - "atr": Uses backward-looking volume profile levels (legacy)

        Args:
            df: Full DataFrame.
            idx: Entry bar index position.
            direction: 1 for long, -1 for short.
            feature_zscore: Optional Z-score of active feature at entry.

        Returns:
            Take profit price.
        """
        entry_price = df["close"].iloc[idx]
        mode = self.params.get("stop_loss_mode", "atr")

        if mode == "fixed":
            zscore_threshold = self.params.get("feature_zscore_threshold", 2.5)
            if feature_zscore is not None and abs(feature_zscore) >= zscore_threshold:
                fixed_points = self.params.get("extended_take_profit_fixed_points", 50)
            else:
                fixed_points = self.params.get("take_profit_fixed_points", 30)
            if direction == 1:
                return entry_price + fixed_points
            else:
                return entry_price - fixed_points
        else:
            # Legacy ATR-based / volume-profile-based logic
            atr = self._compute_atr(df, idx)
            lookback = self.params["take_profit_lookback"]

            # Determine minimum TP distance based on feature Z-score
            if feature_zscore is not None and abs(feature_zscore) >= 2.5:
                min_atr_mult = 2.5
            else:
                min_atr_mult = 1.5

            # Use backward-looking data only (bars up to and including idx)
            start_idx = max(0, idx - lookback)
            segment = df.iloc[start_idx : idx + 1]

            if len(segment) > 1:
                if direction == 1:
                    above_entry = segment[segment["high"] > entry_price]
                    if len(above_entry) > 0:
                        weights = above_entry["volume"].values
                        if weights.sum() > 0:
                            target = np.average(above_entry["high"].values, weights=weights)
                        else:
                            target = above_entry["high"].mean()
                        min_target = entry_price + atr * min_atr_mult
                        return max(target, min_target)
                else:
                    below_entry = segment[segment["low"] < entry_price]
                    if len(below_entry) > 0:
                        weights = below_entry["volume"].values
                        if weights.sum() > 0:
                            target = np.average(below_entry["low"].values, weights=weights)
                        else:
                            target = below_entry["low"].mean()
                        max_target = entry_price - atr * min_atr_mult
                        return min(target, max_target)

            # Fallback: ATR-based target
            return entry_price + direction * atr * 2.0

    def _compute_atr(self, df: pd.DataFrame, idx: int) -> float:
        """Compute ATR at a specific bar."""
        period = self.params["atr_period"]
        start = max(0, idx - period)
        segment = df.iloc[start : idx + 1]

        if len(segment) < 2:
            return df["high"].iloc[idx] - df["low"].iloc[idx]

        tr = pd.concat(
            [
                segment["high"] - segment["low"],
                (segment["high"] - segment["close"].shift(1)).abs(),
                (segment["low"] - segment["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.mean()

    def get_param_ranges(self) -> dict[str, list]:
        """Return parameter ranges for optimization.

        Reduced grid for intraday scalping to keep optimization time
        reasonable with large bar counts.
        """
        return {
            "rotation_min_bars": [5, 10],
            "min_trend_strength": [15.0, 25.0],
        }
