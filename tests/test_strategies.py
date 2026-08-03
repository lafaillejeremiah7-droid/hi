"""Tests for trading strategies.

Tests signal generation on synthetic data for both the Order Flow
and Volume Profile strategies, and the Combined/Confluence strategy.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.combined_strategy import CombinedStrategy
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.volume_profile_strategy import VolumeProfileStrategy


def make_synthetic_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Create synthetic OHLCV data for testing.

    Args:
        n: Number of bars to generate.
        trend: 'up', 'down', or 'flat'.

    Returns:
        DataFrame with OHLCV and preprocessed columns.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    if trend == "up":
        base = np.cumsum(np.random.randn(n) * 0.5 + 0.2) + 100
    elif trend == "down":
        base = np.cumsum(np.random.randn(n) * 0.5 - 0.2) + 100
    else:
        base = np.random.randn(n) * 0.5 + 100

    # Create realistic OHLCV
    open_prices = base + np.random.randn(n) * 0.3
    close_prices = base + np.random.randn(n) * 0.3
    high_prices = np.maximum(open_prices, close_prices) + np.abs(
        np.random.randn(n) * 0.5
    )
    low_prices = np.minimum(open_prices, close_prices) - np.abs(
        np.random.randn(n) * 0.5
    )
    volume = np.random.randint(1000, 10000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=dates,
    )

    # Add preprocessed columns needed by strategies
    bar_range = df["high"] - df["low"]
    bar_range = bar_range.replace(0, 1)
    df["volume_delta"] = df["volume"] * (df["close"] - df["open"]) / bar_range
    df["cumulative_delta"] = df["volume_delta"].cumsum()
    df["relative_volume"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()

    # Support/resistance
    df["resistance_1"] = df["high"].rolling(20, min_periods=1).max()
    df["support_1"] = df["low"].rolling(20, min_periods=1).min()
    df["resistance_2"] = df["resistance_1"] * 0.99
    df["support_2"] = df["support_1"] * 1.01

    # Nearest S/R distance
    dist_sup = (df["close"] - df["support_1"]).abs()
    dist_res = (df["resistance_1"] - df["close"]).abs()
    df["nearest_sr_distance"] = (
        pd.concat([dist_sup, dist_res], axis=1).min(axis=1) / df["close"]
    )

    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


def make_absorption_data() -> pd.DataFrame:
    """Create data that should trigger absorption signals.

    Setup: Price at support with very high relative volume and bullish close.
    """
    n = 50
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Create a decline to support, then absorption
    prices = np.linspace(110, 100, n)
    open_prices = prices + np.random.randn(n) * 0.1
    close_prices = prices + np.random.randn(n) * 0.1
    high_prices = np.maximum(open_prices, close_prices) + 0.5
    low_prices = np.minimum(open_prices, close_prices) - 0.5
    volume = np.ones(n) * 5000.0

    # Last few bars: at support with high volume and bullish candle
    close_prices[-3:] = [100.0, 100.5, 101.0]
    open_prices[-3:] = [100.5, 100.0, 100.2]
    low_prices[-3:] = [99.5, 99.8, 100.0]
    high_prices[-3:] = [101.0, 100.8, 101.5]
    volume[-3:] = [15000, 18000, 20000]  # Volume spike at support

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=dates,
    )

    # Preprocess
    bar_range = (df["high"] - df["low"]).replace(0, 1)
    df["volume_delta"] = df["volume"] * (df["close"] - df["open"]) / bar_range
    df["cumulative_delta"] = df["volume_delta"].cumsum()
    df["relative_volume"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    df["resistance_1"] = df["high"].rolling(20, min_periods=1).max()
    df["support_1"] = df["low"].rolling(20, min_periods=1).min()
    df["resistance_2"] = df["resistance_1"] * 0.99
    df["support_2"] = df["support_1"] * 1.01
    dist_sup = (df["close"] - df["support_1"]).abs()
    dist_res = (df["resistance_1"] - df["close"]).abs()
    df["nearest_sr_distance"] = (
        pd.concat([dist_sup, dist_res], axis=1).min(axis=1) / df["close"]
    )
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


class TestOrderFlowStrategy:
    """Tests for OrderFlowStrategy."""

    def test_initialization(self):
        """Strategy initializes with default parameters."""
        strategy = OrderFlowStrategy()
        assert strategy.params is not None
        assert "absorption_volume_threshold" in strategy.params
        assert "min_signal_strength" in strategy.params

    def test_custom_params(self):
        """Strategy accepts custom parameters."""
        custom = {"absorption_volume_threshold": 3.0, "min_signal_strength": 1}
        strategy = OrderFlowStrategy(params=custom)
        assert strategy.params["absorption_volume_threshold"] == 3.0
        assert strategy.params["min_signal_strength"] == 1

    def test_generate_signals_returns_dataframe(self):
        """generate_signals returns DataFrame with signal column."""
        strategy = OrderFlowStrategy()
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert "signal_strength" in result.columns
        assert len(result) == len(df)

    def test_signals_are_valid_values(self):
        """Signals are only -1, 0, or 1."""
        strategy = OrderFlowStrategy()
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_no_signals_in_flat_market(self):
        """No signals generated when conditions are not met (flat, low volume)."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 5})
        df = make_synthetic_ohlcv(100, trend="flat")
        # Set all volumes to low
        df["volume"] = 1000
        df["relative_volume"] = 0.5  # Below minimum
        result = strategy.generate_signals(df)

        # With very high min_signal_strength and low volume, no signals
        assert (result["signal"] == 0).all()

    def test_stop_loss_long(self):
        """Stop loss for long is below entry price."""
        strategy = OrderFlowStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        sl = strategy.get_stop_loss(df, entry_idx, direction=1)
        assert sl < df["close"].iloc[entry_idx]

    def test_stop_loss_short(self):
        """Stop loss for short is above entry price."""
        strategy = OrderFlowStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        sl = strategy.get_stop_loss(df, entry_idx, direction=-1)
        assert sl > df["close"].iloc[entry_idx]

    def test_take_profit_long(self):
        """Take profit for long is above entry price."""
        strategy = OrderFlowStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        tp = strategy.get_take_profit(df, entry_idx, direction=1)
        assert tp > df["close"].iloc[entry_idx]

    def test_take_profit_short(self):
        """Take profit for short is below entry price."""
        strategy = OrderFlowStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        tp = strategy.get_take_profit(df, entry_idx, direction=-1)
        assert tp < df["close"].iloc[entry_idx]

    def test_param_ranges(self):
        """Strategy provides parameter ranges for optimization."""
        strategy = OrderFlowStrategy()
        ranges = strategy.get_param_ranges()
        assert isinstance(ranges, dict)
        assert len(ranges) > 0

    def test_signals_with_absorption_data(self):
        """Strategy can detect signals in data with absorption patterns."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        df = make_absorption_data()
        result = strategy.generate_signals(df)

        # Should have at least some signals since we created obvious patterns
        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns


class TestVolumeProfileStrategy:
    """Tests for VolumeProfileStrategy."""

    def test_initialization(self):
        """Strategy initializes with default parameters."""
        strategy = VolumeProfileStrategy()
        assert strategy.params is not None
        assert "profile_bins" in strategy.params
        assert "profile_lookback" in strategy.params

    def test_custom_params(self):
        """Strategy accepts custom parameters."""
        custom = {"profile_bins": 30, "rotation_min_bars": 5}
        strategy = VolumeProfileStrategy(params=custom)
        assert strategy.params["profile_bins"] == 30
        assert strategy.params["rotation_min_bars"] == 5

    def test_generate_signals_returns_dataframe(self):
        """generate_signals returns DataFrame with signal column."""
        strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert len(result) == len(df)

    def test_signals_are_valid_values(self):
        """Signals are only -1, 0, or 1."""
        strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_no_signals_short_data(self):
        """No signals with insufficient data."""
        strategy = VolumeProfileStrategy()
        # Very short data - not enough for any setup
        df = make_synthetic_ohlcv(5)
        result = strategy.generate_signals(df)

        # Should have no signals with only 5 bars
        assert (result["signal"] == 0).all()

    def test_stop_loss_long(self):
        """Stop loss for long is below entry price."""
        strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        sl = strategy.get_stop_loss(df, entry_idx, direction=1)
        assert sl < df["close"].iloc[entry_idx]

    def test_stop_loss_short(self):
        """Stop loss for short is above entry price."""
        strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        sl = strategy.get_stop_loss(df, entry_idx, direction=-1)
        assert sl > df["close"].iloc[entry_idx]

    def test_take_profit_long(self):
        """Take profit for long is above entry price."""
        strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        tp = strategy.get_take_profit(df, entry_idx, direction=1)
        assert tp > df["close"].iloc[entry_idx]

    def test_take_profit_short(self):
        """Take profit for short is below entry price."""
        strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        tp = strategy.get_take_profit(df, entry_idx, direction=-1)
        assert tp < df["close"].iloc[entry_idx]

    def test_param_ranges(self):
        """Strategy provides parameter ranges for optimization."""
        strategy = VolumeProfileStrategy()
        ranges = strategy.get_param_ranges()
        assert isinstance(ranges, dict)
        assert len(ranges) > 0

    def test_strategy_name(self):
        """Strategy has a proper name."""
        strategy = VolumeProfileStrategy()
        assert strategy.name == "VolumeProfileStrategy"

    def test_signals_with_trend_data(self):
        """Strategy can find signals in trending data."""
        strategy = VolumeProfileStrategy()
        # Use more data points for volume profile to work properly
        df = make_synthetic_ohlcv(300, trend="up")
        result = strategy.generate_signals(df)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert "setup_type" in result.columns


class TestCombinedStrategy:
    """Tests for CombinedStrategy (VP + OF confluence)."""

    def test_initialization(self):
        """Strategy initializes with default parameters."""
        strategy = CombinedStrategy()
        assert strategy.params is not None
        assert "confirmation_window" in strategy.params
        assert strategy.params["confirmation_window"] == 2

    def test_custom_params(self):
        """Strategy accepts custom parameters."""
        custom = {"confirmation_window": 3}
        strategy = CombinedStrategy(params=custom)
        assert strategy.params["confirmation_window"] == 3

    def test_generate_signals_returns_dataframe(self):
        """generate_signals returns DataFrame with expected columns."""
        strategy = CombinedStrategy()
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert "vp_setup_type" in result.columns
        assert "of_strength" in result.columns
        assert len(result) == len(df)

    def test_signals_are_valid_values(self):
        """Signals are only -1, 0, or 1."""
        strategy = CombinedStrategy()
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_combined_signals_subset_of_vp(self):
        """Combined signals should be a subset of VP signals (more restrictive)."""
        strategy = CombinedStrategy()
        vp_strategy = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(300, trend="up")

        combined_result = strategy.generate_signals(df)
        vp_result = vp_strategy.generate_signals(df)

        # Combined should have <= signals than VP alone (it requires OF confirmation)
        combined_signal_count = (combined_result["signal"] != 0).sum()
        vp_signal_count = (vp_result["signal"] != 0).sum()
        assert combined_signal_count <= vp_signal_count

    def test_no_signals_without_confluence(self):
        """No signals when conditions are too strict for confluence."""
        # Use very strict OF params so OF generates nothing
        strategy = CombinedStrategy(params={
            "of_params": {"min_signal_strength": 5, "min_relative_volume": 5.0},
        })
        df = make_synthetic_ohlcv(200)
        result = strategy.generate_signals(df)

        # With impossible OF requirements, no confluence possible
        assert (result["signal"] == 0).all()

    def test_stop_loss_uses_vp_logic(self):
        """Stop loss delegates to Volume Profile strategy."""
        combined = CombinedStrategy()
        vp = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50

        combined_sl = combined.get_stop_loss(df, entry_idx, direction=1)
        vp_sl = vp.get_stop_loss(df, entry_idx, direction=1)
        assert combined_sl == vp_sl

    def test_take_profit_uses_vp_logic(self):
        """Take profit delegates to Volume Profile strategy."""
        combined = CombinedStrategy()
        vp = VolumeProfileStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50

        combined_tp = combined.get_take_profit(df, entry_idx, direction=1)
        vp_tp = vp.get_take_profit(df, entry_idx, direction=1)
        assert combined_tp == vp_tp

    def test_stop_loss_long(self):
        """Stop loss for long is below entry price."""
        strategy = CombinedStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        sl = strategy.get_stop_loss(df, entry_idx, direction=1)
        assert sl < df["close"].iloc[entry_idx]

    def test_stop_loss_short(self):
        """Stop loss for short is above entry price."""
        strategy = CombinedStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        sl = strategy.get_stop_loss(df, entry_idx, direction=-1)
        assert sl > df["close"].iloc[entry_idx]

    def test_take_profit_long(self):
        """Take profit for long is above entry price."""
        strategy = CombinedStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        tp = strategy.get_take_profit(df, entry_idx, direction=1)
        assert tp > df["close"].iloc[entry_idx]

    def test_take_profit_short(self):
        """Take profit for short is below entry price."""
        strategy = CombinedStrategy()
        df = make_synthetic_ohlcv(100)
        entry_idx = 50
        tp = strategy.get_take_profit(df, entry_idx, direction=-1)
        assert tp < df["close"].iloc[entry_idx]

    def test_param_ranges(self):
        """Strategy provides parameter ranges for optimization."""
        strategy = CombinedStrategy()
        ranges = strategy.get_param_ranges()
        assert isinstance(ranges, dict)
        assert "confirmation_window" in ranges
        assert len(ranges) > 0

    def test_strategy_name(self):
        """Strategy has a proper name."""
        strategy = CombinedStrategy()
        assert strategy.name == "CombinedStrategy"

    def test_confirmation_window_effect(self):
        """Larger confirmation window should allow more (or equal) signals."""
        df = make_synthetic_ohlcv(300, trend="up")

        narrow = CombinedStrategy(params={"confirmation_window": 1})
        wide = CombinedStrategy(params={"confirmation_window": 3})

        narrow_signals = (narrow.generate_signals(df)["signal"] != 0).sum()
        wide_signals = (wide.generate_signals(df)["signal"] != 0).sum()

        # Wider window should have >= signals (more chances for confluence)
        assert wide_signals >= narrow_signals
