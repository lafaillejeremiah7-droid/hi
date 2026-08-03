"""Combined/Confluence Strategy implementation.

Requires BOTH Volume Profile level identification AND Order Flow confirmation
before entering a trade. This mimics professional order flow trading:
1. Volume Profile identifies WHERE to trade (heavy volume zones)
2. Order Flow provides WHEN to enter (absorption, delta divergence, etc.)
3. Entry only when both agree on direction within a confirmation window.

Supports scalping parameters: max_trades_per_day, max_hold_bars,
and trading session time filters (handled by the engine).
"""

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.volume_profile_strategy import VolumeProfileStrategy


class CombinedStrategy(BaseStrategy):
    """Combined strategy requiring VP level + OF confirmation.

    Uses VolumeProfileStrategy to identify valid levels/zones, then
    requires OrderFlowStrategy signals to confirm direction at those
    levels within a configurable confirmation window.

    Stop loss and take profit use Volume Profile logic (low-volume area
    behind heavy volume zone, next heavy volume zone as target).

    Scalping parameters (max_trades_per_day, max_hold_bars,
    trading_session_start/end) are passed to the engine, not
    handled by signal generation directly.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """Initialize combined strategy with sub-strategies.

        Args:
            params: Combined strategy parameters. Can include
                    vp_params and of_params dicts to configure sub-strategies.
        """
        super().__init__(params)

        # Extract sub-strategy params if provided
        vp_params = self.params.get("vp_params", None)
        of_params = self.params.get("of_params", None)

        self._vp_strategy = VolumeProfileStrategy(params=vp_params)
        self._of_strategy = OrderFlowStrategy(params=of_params)

    def default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            # How many bars the OF signal can lag/lead the VP signal
            "confirmation_window": 2,
            # Max trades per day (scalping constraint, used by engine)
            "max_trades_per_day": 2,
            # Trading session filters (used by engine)
            "trading_session_start": "09:30",
            "trading_session_end": "16:00",
            "session_timezone": "US/Eastern",
            # Use VP stop loss and take profit logic
            "stop_loss_atr_mult": 1.0,
            "take_profit_lookback": 50,
            "atr_period": 10,
            # Sub-strategy params (None means use their defaults)
            "vp_params": None,
            "of_params": None,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals requiring VP + OF confluence (vectorized).

        A signal is generated only when:
        1. Volume Profile generates a signal (level identified)
        2. Order Flow generates a signal in the same direction
        3. Both signals occur within the confirmation_window

        Uses rolling window operations instead of per-bar loops for
        performance on large datasets (100K+ bars).

        Args:
            df: Preprocessed DataFrame with OHLCV and indicator columns.

        Returns:
            DataFrame with 'signal', 'vp_setup_type', and 'of_strength' columns.
        """
        window = self.params["confirmation_window"]

        # Generate signals from both sub-strategies
        vp_signals = self._vp_strategy.generate_signals(df)
        of_signals = self._of_strategy.generate_signals(df)

        vp_sig = vp_signals["signal"]
        of_sig = of_signals["signal"]

        # Vectorized confluence detection using rolling windows
        # For each direction, check if both VP and OF have signals within window
        win_size = 2 * window + 1  # Full window centered on each bar

        # Create rolling indicators for presence of signals within window
        # VP long/short presence in rolling window
        vp_long_indicator = (vp_sig == 1).astype(int)
        vp_short_indicator = (vp_sig == -1).astype(int)
        of_long_indicator = (of_sig == 1).astype(int)
        of_short_indicator = (of_sig == -1).astype(int)

        # Use rolling sum to detect if any signal exists within the window
        # We need a centered window, so we use forward + backward rolling
        vp_long_window = vp_long_indicator.rolling(
            window=win_size, min_periods=1, center=True
        ).sum()
        vp_short_window = vp_short_indicator.rolling(
            window=win_size, min_periods=1, center=True
        ).sum()
        of_long_window = of_long_indicator.rolling(
            window=win_size, min_periods=1, center=True
        ).sum()
        of_short_window = of_short_indicator.rolling(
            window=win_size, min_periods=1, center=True
        ).sum()

        # Confluence: both VP and OF have signals in same direction within window
        long_confluence = (vp_long_window > 0) & (of_long_window > 0)
        short_confluence = (vp_short_window > 0) & (of_short_window > 0)

        # For each confluence zone, fire signal at the bar where the
        # confirming signal arrives (the latest of VP and OF)
        # Approximate: fire at the bar where either VP or OF actually fires
        # and the other is present within the window
        signal = pd.Series(0, index=df.index, dtype=int)

        # Long signals: fire where OF has a signal AND VP is within window
        # OR where VP has a signal AND OF is within window
        # Pick the latest trigger (approximated by requiring actual signal at bar)
        long_trigger = long_confluence & (
            (vp_sig == 1) | (of_sig == 1)
        )
        short_trigger = short_confluence & (
            (vp_sig == -1) | (of_sig == -1)
        )

        # Deduplicate: only fire once per confluence zone
        # Use diff to find first occurrence in each cluster
        long_groups = (~long_trigger).cumsum()
        short_groups = (~short_trigger).cumsum()

        # Within each confluence group, only keep the first signal
        long_first = long_trigger & (~long_trigger.shift(1, fill_value=False))
        short_first = short_trigger & (~short_trigger.shift(1, fill_value=False))

        signal[long_first] = 1
        signal[short_first & (signal == 0)] = -1

        # Build result DataFrame
        result = pd.DataFrame(index=df.index)
        result["signal"] = signal

        # Map VP setup types and OF strengths for signal bars
        vp_setup_type = pd.Series("", index=df.index, dtype=str)
        of_strength = pd.Series(0, index=df.index, dtype=int)

        if "setup_type" in vp_signals.columns:
            # For signal bars, find the nearest VP signal's setup_type
            vp_setup = vp_signals["setup_type"]
            # Forward-fill VP setup type with limit=window to carry it through window
            vp_setup_filled = vp_setup.where(vp_sig != 0).ffill(limit=window)
            vp_setup_type = vp_setup_filled.where(signal != 0, "").fillna("")

        if "signal_strength" in of_signals.columns:
            of_str = of_signals["signal_strength"]
            of_str_filled = of_str.where(of_sig != 0).ffill(limit=window)
            of_strength = of_str_filled.where(signal != 0, 0).fillna(0).astype(int)

        result["vp_setup_type"] = vp_setup_type
        result["of_strength"] = of_strength

        return result

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Calculate stop loss using Volume Profile logic.

        Places stop loss in low-volume area behind the heavy volume zone,
        using ATR-based placement from the VP strategy.

        Args:
            df: Full DataFrame.
            idx: Entry bar index position.
            direction: 1 for long, -1 for short.

        Returns:
            Stop loss price.
        """
        return self._vp_strategy.get_stop_loss(df, idx, direction)

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int, feature_zscore: float | None = None) -> float:
        """Calculate take profit using Volume Profile logic.

        Uses the next heavy volume zone as the target, following
        VP strategy methodology. Passes feature_zscore for dynamic TP.

        Args:
            df: Full DataFrame.
            idx: Entry bar index position.
            direction: 1 for long, -1 for short.
            feature_zscore: Optional Z-score of active feature at entry.

        Returns:
            Take profit price.
        """
        return self._vp_strategy.get_take_profit(df, idx, direction, feature_zscore=feature_zscore)

    def get_param_ranges(self) -> dict[str, list]:
        """Return parameter ranges for optimization.

        Minimal grid (2 values) to keep walk-forward optimization fast.
        """
        return {
            "confirmation_window": [2, 3],
        }
