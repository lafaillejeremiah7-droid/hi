"""Order Flow Strategy implementation.

Uses REAL order flow data (bid_volume, ask_volume, delta) when available
for accurate signal generation. Falls back to OHLCV proxies when real
data is not present.

Real order flow signals:
1. REAL Absorption: both sides heavy at S/R zones
2. REAL Cumulative Delta Divergence: price vs delta direction mismatch
3. REAL Imbalance: one side dominates heavily (3x ratio)
4. Trapped Traders: heavy volume at extremes followed by reversal
5. Failed Auctions: incomplete auction at bar extremes
"""

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class OrderFlowStrategy(BaseStrategy):
    """Order flow strategy using real bid/ask data or OHLCV proxies.

    When real order flow columns (bid_volume, ask_volume, delta,
    absorption_signal, imbalance_signal) are present in the data,
    uses them directly for accurate signal generation.

    Falls back to proxy-based indicators when real data is unavailable.
    """

    def default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "absorption_volume_threshold": 2.0,
            "delta_divergence_lookback": 12,
            "imbalance_ratio": 3.0,
            "min_stacked_bars": 3,
            "failed_auction_threshold": 0.1,
            "trapped_lookback": 6,
            "trapped_extreme_volume_pct": 0.3,
            "trapped_volume_threshold": 1.5,
            "sr_zone_width": 1.5,
            "stop_loss_atr_mult": 1.0,
            "take_profit_atr_mult": 1.5,
            "atr_period": 14,
            "min_relative_volume": 0.8,
            "min_signal_strength": 1,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from order flow data.

        Uses real order flow signals when available (absorption_signal,
        imbalance_signal, delta columns). Falls back to proxy indicators.

        Args:
            df: Preprocessed DataFrame with OHLCV and indicator columns.

        Returns:
            DataFrame with 'signal' and 'signal_strength' columns.
        """
        params = self.params
        has_real_of = all(
            col in df.columns for col in ["bid_volume", "ask_volume", "delta"]
        )

        if has_real_of:
            return self._generate_real_signals(df, params)
        else:
            return self._generate_proxy_signals(df, params)

    def _generate_real_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Generate signals using REAL order flow data.

        Sub-signals:
        1. Absorption: precomputed absorption_signal column
        2. Cumulative delta divergence (real delta)
        3. Imbalance: precomputed imbalance_signal column
        4. Trapped traders (using real bid/ask at bar extremes)
        5. Stacked delta (consecutive bars with same-direction imbalance)

        Args:
            df: DataFrame with real order flow columns.
            params: Strategy parameters.

        Returns:
            DataFrame with signal and signal_strength columns.
        """
        result = pd.DataFrame(index=df.index)

        # Sub-signal 1: Absorption (from preprocessor)
        if "absorption_signal" in df.columns:
            absorption = df["absorption_signal"].astype(int)
        else:
            absorption = self._compute_real_absorption(df, params)

        # Sub-signal 2: Cumulative delta divergence
        divergence = self._compute_real_delta_divergence(df, params)

        # Sub-signal 3: Imbalance (from preprocessor or computed here)
        if "imbalance_signal" in df.columns:
            imbalance = df["imbalance_signal"].astype(int)
        else:
            imbalance = self._compute_real_imbalance(df, params)

        # Sub-signal 4: Trapped traders (real version)
        trapped = self._compute_real_trapped_traders(df, params)

        # Sub-signal 5: Stacked imbalances (consecutive same-direction delta)
        stacked = self._compute_stacked_imbalances(df, params)

        # Combine sub-signals
        bullish_count = (
            (absorption == 1).astype(int)
            + (divergence == 1).astype(int)
            + (imbalance == 1).astype(int)
            + (trapped == 1).astype(int)
            + (stacked == 1).astype(int)
        )

        bearish_count = (
            (absorption == -1).astype(int)
            + (divergence == -1).astype(int)
            + (imbalance == -1).astype(int)
            + (trapped == -1).astype(int)
            + (stacked == -1).astype(int)
        )

        # S/R zone filter
        at_sr_zone = df["nearest_sr_distance"] < (params["sr_zone_width"] / 100)

        # Volume filter
        vol_filter = df["relative_volume"] >= params["min_relative_volume"]

        # Generate final signal
        min_strength = params["min_signal_strength"]
        signal = pd.Series(0, index=df.index, dtype=int)

        bullish_valid = (bullish_count >= min_strength) & at_sr_zone & vol_filter
        bearish_valid = (bearish_count >= min_strength) & at_sr_zone & vol_filter

        signal[bullish_valid] = 1
        signal[bearish_valid] = -1

        # Resolve conflicts
        both_valid = bullish_valid & bearish_valid
        signal[both_valid & (bullish_count > bearish_count)] = 1
        signal[both_valid & (bearish_count > bullish_count)] = -1
        signal[both_valid & (bullish_count == bearish_count)] = 0

        result["signal"] = signal
        result["signal_strength"] = pd.concat(
            [bullish_count, bearish_count], axis=1
        ).max(axis=1)

        return result

    def _compute_real_absorption(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute real absorption signal from bid/ask volume.

        Both bid_volume AND ask_volume must be > threshold * their
        rolling average simultaneously.

        Args:
            df: DataFrame with bid_volume, ask_volume.
            params: Strategy parameters.

        Returns:
            Series: +1 bullish, -1 bearish, 0 none.
        """
        result = pd.Series(0, index=df.index, dtype=int)
        threshold = params["absorption_volume_threshold"]

        bid_vol = df["bid_volume"].astype(float)
        ask_vol = df["ask_volume"].astype(float)

        bid_avg = bid_vol.rolling(window=20, min_periods=1).mean()
        ask_avg = ask_vol.rolling(window=20, min_periods=1).mean()

        absorption = (bid_vol > bid_avg * threshold) & (ask_vol > ask_avg * threshold)

        # Direction from S/R proximity
        if "support_1" in df.columns and "resistance_1" in df.columns:
            dist_to_support = (df["close"] - df["support_1"]).abs()
            dist_to_resistance = (df["resistance_1"] - df["close"]).abs()
            near_support = dist_to_support < dist_to_resistance
            result[absorption & near_support] = 1
            result[absorption & ~near_support] = -1

        return result

    def _compute_real_delta_divergence(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute real cumulative delta divergence.

        Divergence: price making new highs but cumulative delta declining,
        or price making new lows but cumulative delta rising.

        Uses REAL cumulative delta (sum of real bid-ask delta).

        Args:
            df: DataFrame with close, cumulative_delta.
            params: Strategy parameters.

        Returns:
            Series: +1 bullish divergence, -1 bearish divergence, 0 none.
        """
        result = pd.Series(0, index=df.index, dtype=int)
        lookback = params["delta_divergence_lookback"]

        if len(df) < lookback + 1:
            return result

        close = df["close"]
        cum_delta = df["cumulative_delta"]

        # Price at rolling highs, delta declining
        price_high = close == close.rolling(window=lookback, min_periods=lookback).max()
        delta_declining = cum_delta < cum_delta.shift(lookback)

        # Price at rolling lows, delta rising
        price_low = close == close.rolling(window=lookback, min_periods=lookback).min()
        delta_rising = cum_delta > cum_delta.shift(lookback)

        # Bearish divergence: price high + delta declining
        result[price_high & delta_declining] = -1
        # Bullish divergence: price low + delta rising
        result[price_low & delta_rising] = 1

        return result

    def _compute_real_imbalance(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute real imbalance from bid/ask ratio.

        Args:
            df: DataFrame with bid_volume, ask_volume.
            params: Strategy parameters.

        Returns:
            Series: +1 buying imbalance, -1 selling imbalance, 0 balanced.
        """
        result = pd.Series(0, index=df.index, dtype=int)
        ratio = params["imbalance_ratio"]

        bid_vol = df["bid_volume"].astype(float)
        ask_vol = df["ask_volume"].astype(float)

        safe_ask = ask_vol.replace(0, 1)
        safe_bid = bid_vol.replace(0, 1)

        result[bid_vol / safe_ask >= ratio] = 1
        result[ask_vol / safe_bid >= ratio] = -1

        return result

    def _compute_real_trapped_traders(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Detect trapped traders using real bid/ask volume.

        Trapped buyers: heavy bid_volume (buying) on a bar, but the
        next bar is bearish. Buyers got trapped at highs.

        Trapped sellers: heavy ask_volume (selling) on a bar, but the
        next bar is bullish. Sellers got trapped at lows.

        Args:
            df: DataFrame with bid_volume, ask_volume, OHLC.
            params: Strategy parameters.

        Returns:
            Series: +1 trapped sellers (go long), -1 trapped buyers (go short).
        """
        result = pd.Series(0, index=df.index, dtype=int)
        threshold = params["trapped_volume_threshold"]

        if len(df) < 2:
            return result

        bid_vol = df["bid_volume"].astype(float)
        ask_vol = df["ask_volume"].astype(float)

        bid_avg = bid_vol.rolling(window=20, min_periods=1).mean()
        ask_avg = ask_vol.rolling(window=20, min_periods=1).mean()

        # Strong buying bar (high bid volume) at bar highs
        strong_buying = (bid_vol > bid_avg * threshold) & (df["close"] > df["open"])
        # Strong selling bar (high ask volume) at bar lows
        strong_selling = (ask_vol > ask_avg * threshold) & (df["close"] < df["open"])

        # Next bar reverses direction
        next_bearish = df["close"].shift(-1) < df["open"].shift(-1)
        next_bullish = df["close"].shift(-1) > df["open"].shift(-1)

        # Trapped buyers: strong buying then reversal
        # Signal on the reversal bar (shift forward by 1)
        trapped_buyers = (strong_buying & next_bearish).shift(1).fillna(False)
        result[trapped_buyers] = -1

        # Trapped sellers: strong selling then reversal
        trapped_sellers = (strong_selling & next_bullish).shift(1).fillna(False)
        result[trapped_sellers] = 1

        return result

    def _compute_stacked_imbalances(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Detect stacked imbalances (consecutive same-direction delta).

        When delta is positive for min_stacked_bars consecutive bars,
        that creates buying pressure (support). Vice versa for selling.

        Args:
            df: DataFrame with delta column.
            params: Strategy parameters.

        Returns:
            Series: +1 stacked buying, -1 stacked selling, 0 none.
        """
        result = pd.Series(0, index=df.index, dtype=int)
        min_bars = params["min_stacked_bars"]

        if len(df) < min_bars:
            return result

        delta = df["delta"].astype(float) if "delta" in df.columns else df["volume_delta"]

        # Count consecutive positive/negative delta bars
        positive = (delta > 0).astype(int)
        negative = (delta < 0).astype(int)

        # Rolling sum over min_bars window
        pos_sum = positive.rolling(window=min_bars, min_periods=min_bars).sum()
        neg_sum = negative.rolling(window=min_bars, min_periods=min_bars).sum()

        # Stacked: all bars in window have same direction
        result[pos_sum == min_bars] = 1
        result[neg_sum == min_bars] = -1

        return result

    def _generate_proxy_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Generate signals using OHLCV proxy indicators (fallback).

        Used when real order flow data is not available.

        Args:
            df: DataFrame with OHLCV and proxy indicator columns.
            params: Strategy parameters.

        Returns:
            DataFrame with signal and signal_strength columns.
        """
        from src.indicators.order_flow import (
            cumulative_delta_divergence,
            detect_absorption,
            detect_failed_auctions,
            detect_stacked_imbalances,
            detect_trapped_traders,
        )

        result = pd.DataFrame(index=df.index)

        absorption = detect_absorption(
            df, threshold=params["absorption_volume_threshold"]
        )
        divergence = cumulative_delta_divergence(
            df, lookback=params["delta_divergence_lookback"]
        )
        imbalances = detect_stacked_imbalances(
            df,
            ratio=params["imbalance_ratio"],
            min_bars=params["min_stacked_bars"],
        )
        failed_auctions = detect_failed_auctions(
            df, threshold=params["failed_auction_threshold"]
        )
        trapped = detect_trapped_traders(
            df, lookback=params["trapped_lookback"]
        )

        bullish_count = (
            (absorption == 1).astype(int)
            + (divergence == 1).astype(int)
            + (imbalances == 1).astype(int)
            + (failed_auctions == 1).astype(int)
            + (trapped == 1).astype(int)
        )

        bearish_count = (
            (absorption == -1).astype(int)
            + (divergence == -1).astype(int)
            + (imbalances == -1).astype(int)
            + (failed_auctions == -1).astype(int)
            + (trapped == -1).astype(int)
        )

        at_sr_zone = df["nearest_sr_distance"] < (params["sr_zone_width"] / 100)
        vol_filter = df["relative_volume"] >= params["min_relative_volume"]

        min_strength = params["min_signal_strength"]
        signal = pd.Series(0, index=df.index, dtype=int)

        bullish_valid = (bullish_count >= min_strength) & at_sr_zone & vol_filter
        bearish_valid = (bearish_count >= min_strength) & at_sr_zone & vol_filter

        signal[bullish_valid] = 1
        signal[bearish_valid] = -1

        both_valid = bullish_valid & bearish_valid
        signal[both_valid & (bullish_count > bearish_count)] = 1
        signal[both_valid & (bearish_count > bullish_count)] = -1
        signal[both_valid & (bullish_count == bearish_count)] = 0

        result["signal"] = signal
        result["signal_strength"] = pd.concat(
            [bullish_count, bearish_count], axis=1
        ).max(axis=1)

        return result

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Calculate ATR-based stop loss.

        Args:
            df: Full DataFrame with data.
            idx: Entry bar index position.
            direction: 1 for long, -1 for short.

        Returns:
            Stop loss price.
        """
        atr = self._compute_atr(df, idx)
        entry_price = df["close"].iloc[idx]
        mult = self.params["stop_loss_atr_mult"]

        if direction == 1:
            return entry_price - atr * mult
        else:
            return entry_price + atr * mult

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int, feature_zscore: float | None = None) -> float:
        """Calculate ATR-based take profit.

        If |feature_zscore| >= 2.5, uses extended TP multiplier (2.5x ATR).
        Otherwise uses default TP multiplier (1.5x ATR).

        Args:
            df: Full DataFrame with data.
            idx: Entry bar index position.
            direction: 1 for long, -1 for short.
            feature_zscore: Optional Z-score of active feature at entry.

        Returns:
            Take profit price.
        """
        atr = self._compute_atr(df, idx)
        entry_price = df["close"].iloc[idx]

        # Determine TP multiplier based on feature Z-score
        if feature_zscore is not None and abs(feature_zscore) >= 2.5:
            mult = 2.5
        else:
            mult = self.params["take_profit_atr_mult"]

        if direction == 1:
            return entry_price + atr * mult
        else:
            return entry_price - atr * mult

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
        """Return parameter ranges for optimization."""
        return {
            "min_signal_strength": [1, 2],
            "sr_zone_width": [1.5, 2.0],
            "min_relative_volume": [0.5, 0.8],
        }
