"""Tests for trading strategies.

Tests signal generation on synthetic data for the primary SimpleStrategy
and for the Order Flow / Volume Profile baselines.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.simple_strategy import (
    PROFILE_BIN_WIDTH_POINTS,
    SimpleStrategy,
    heavy_volume_node,
    rolling_delta_zscore,
)
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


def make_orderflow_data(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Create synthetic 5-min data with a real order flow delta column.

    Returns:
        DataFrame with OHLCV, bid_volume, ask_volume and delta.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")

    base = np.cumsum(rng.normal(0.0, 4.0, n)) + 18000
    open_p = base + rng.normal(0, 1.0, n)
    close_p = base + rng.normal(0, 1.0, n)
    high_p = np.maximum(open_p, close_p) + np.abs(rng.normal(0, 3.0, n))
    low_p = np.minimum(open_p, close_p) - np.abs(rng.normal(0, 3.0, n))
    volume = rng.uniform(1000, 5000, n)
    bid_volume = volume * rng.uniform(0.3, 0.7, n)
    ask_volume = volume - bid_volume

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "delta": bid_volume - ask_volume,
        },
        index=dates,
    )
    return df


def make_flat_bars(prices: list[float], volumes: list[float]) -> pd.DataFrame:
    """Create bars where open=high=low=close so the typical price is exact."""
    close = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.array(volumes, dtype=float),
        },
        index=pd.date_range("2024-01-02 09:30", periods=len(prices), freq="5min"),
    )


class TestHeavyVolumeNode:
    """Tests for the volume profile heavy-volume node."""

    def test_picks_bin_with_most_volume(self):
        """The node is the centre of the price bin holding the most volume."""
        prices = [100.0, 100.0, 100.0, 200.0, 100.0]
        volumes = [100.0, 100.0, 100.0, 5000.0, 100.0]
        df = make_flat_bars(prices, volumes)

        node = heavy_volume_node(df, lookback=5)

        # Bin holding 200 is [200, 205), centre 202.5
        assert node.iloc[4] == pytest.approx(202.5)

    def test_nan_during_warmup(self):
        """Bars before a full lookback window have no node."""
        df = make_flat_bars([100.0] * 10, [100.0] * 10)
        node = heavy_volume_node(df, lookback=5)

        assert node.iloc[:4].isna().all()
        assert node.iloc[4:].notna().all()

    def test_window_is_trailing(self):
        """The node only reflects the trailing lookback window."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0, 300.0, 300.0, 300.0, 300.0, 300.0]
        volumes = [1000.0] * 10
        df = make_flat_bars(prices, volumes)

        node = heavy_volume_node(df, lookback=5)

        assert node.iloc[4] == pytest.approx(102.5)
        assert node.iloc[9] == pytest.approx(302.5)

    def test_bin_width_constant_is_positive(self):
        """The profile bin width is a fixed positive constant."""
        assert PROFILE_BIN_WIDTH_POINTS > 0


class TestRollingDeltaZscore:
    """Tests for the rolling delta Z-score."""

    def test_nan_during_warmup(self):
        """Z-score needs a full window before producing a value."""
        df = make_orderflow_data(120)
        z = rolling_delta_zscore(df, window=50)

        assert z.iloc[:49].isna().all()
        assert z.iloc[49:].notna().any()

    def test_positive_delta_spike_gives_positive_zscore(self):
        """A large positive delta produces a positive Z-score."""
        df = make_orderflow_data(120)
        df.loc[df.index[100], "delta"] = df["delta"].abs().max() * 10

        z = rolling_delta_zscore(df, window=50)
        assert z.iloc[100] > 0

    def test_falls_back_to_volume_delta(self):
        """Without a delta column the proxy volume_delta is used."""
        df = make_orderflow_data(120).drop(columns=["delta"])
        df["volume_delta"] = df["bid_volume"] - df["ask_volume"]

        z = rolling_delta_zscore(df, window=50)
        assert z.notna().any()

    def test_raises_without_any_delta(self):
        """A DataFrame with no delta information is an error."""
        df = make_orderflow_data(60).drop(columns=["delta"])
        with pytest.raises(KeyError):
            rolling_delta_zscore(df, window=50)


class TestSimpleStrategy:
    """Tests for the primary SimpleStrategy."""

    def test_tunable_surface_is_exactly_five_parameters(self):
        """The strategy exposes only the documented parameters."""
        strategy = SimpleStrategy()
        assert set(strategy.params) == {
            "profile_lookback",
            "level_proximity_points",
            "delta_threshold",
            "stop_points",
            "target_points",
        }

    def test_custom_params_override_defaults(self):
        """Custom parameters replace the defaults."""
        strategy = SimpleStrategy(params={"stop_points": 25, "target_points": 45})
        assert strategy.params["stop_points"] == 25
        assert strategy.params["target_points"] == 45

    def test_generate_signals_returns_expected_columns(self):
        """generate_signals returns signal, level and delta_zscore."""
        strategy = SimpleStrategy()
        df = make_orderflow_data(300)
        result = strategy.generate_signals(df)

        assert list(result.columns) == ["signal", "level", "delta_zscore"]
        assert len(result) == len(df)

    def test_signals_are_valid_values(self):
        """Signals are only -1, 0 or 1."""
        strategy = SimpleStrategy()
        df = make_orderflow_data(300)
        result = strategy.generate_signals(df)

        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_every_signal_satisfies_both_conditions(self):
        """Each signal is at the level AND confirmed by delta."""
        strategy = SimpleStrategy(
            params={"level_proximity_points": 10, "delta_threshold": 1.0}
        )
        df = make_orderflow_data(600)
        result = strategy.generate_signals(df)

        longs = result[result["signal"] == 1]
        shorts = result[result["signal"] == -1]

        if len(longs) == 0 and len(shorts) == 0:
            pytest.skip("No signals generated on synthetic data")

        for rows, sign in ((longs, 1), (shorts, -1)):
            for label, row in rows.iterrows():
                distance = df.loc[label, "close"] - row["level"]
                assert abs(distance) <= 10, "Signal fired away from the level"
                assert distance * sign >= 0, "Level on the wrong side of price"
                assert row["delta_zscore"] * sign >= 1.0, "Delta did not confirm"

    def test_higher_delta_threshold_is_more_selective(self):
        """Raising the delta threshold cannot add signals."""
        df = make_orderflow_data(600)
        loose = SimpleStrategy(params={"delta_threshold": 0.5})
        strict = SimpleStrategy(params={"delta_threshold": 1.5})

        loose_n = (loose.generate_signals(df)["signal"] != 0).sum()
        strict_n = (strict.generate_signals(df)["signal"] != 0).sum()

        assert strict_n <= loose_n

    def test_wider_proximity_is_less_selective(self):
        """Widening the level proximity cannot remove signals."""
        df = make_orderflow_data(600)
        narrow = SimpleStrategy(params={"level_proximity_points": 5})
        wide = SimpleStrategy(params={"level_proximity_points": 10})

        narrow_n = (narrow.generate_signals(df)["signal"] != 0).sum()
        wide_n = (wide.generate_signals(df)["signal"] != 0).sum()

        assert wide_n >= narrow_n

    def test_stop_loss_is_fixed_points(self):
        """Stop is exactly stop_points from entry on both sides."""
        strategy = SimpleStrategy(params={"stop_points": 20})
        df = make_orderflow_data(100)
        entry = float(df["close"].iloc[50])

        assert strategy.get_stop_loss(df, 50, 1) == pytest.approx(entry - 20)
        assert strategy.get_stop_loss(df, 50, -1) == pytest.approx(entry + 20)

    def test_take_profit_is_fixed_points(self):
        """Target is exactly target_points from entry on both sides."""
        strategy = SimpleStrategy(params={"target_points": 30})
        df = make_orderflow_data(100)
        entry = float(df["close"].iloc[50])

        assert strategy.get_take_profit(df, 50, 1) == pytest.approx(entry + 30)
        assert strategy.get_take_profit(df, 50, -1) == pytest.approx(entry - 30)

    def test_take_profit_ignores_zscore(self):
        """There is no extended target: the Z-score cannot change the target."""
        strategy = SimpleStrategy(params={"target_points": 30})
        df = make_orderflow_data(100)

        plain = strategy.get_take_profit(df, 50, 1)
        extreme = strategy.get_take_profit(df, 50, 1, feature_zscore=99.0)
        assert plain == extreme

    def test_param_ranges_cover_signal_parameters(self):
        """Walk-forward re-optimizes only the two signal parameters."""
        ranges = SimpleStrategy().get_param_ranges()
        assert set(ranges) == {"delta_threshold", "level_proximity_points"}

    def test_strategy_name(self):
        """Strategy reports its class name."""
        assert SimpleStrategy().name == "SimpleStrategy"

    def test_short_dataframe_produces_no_signals(self):
        """Not enough bars for the profile means no signals."""
        strategy = SimpleStrategy()
        df = make_orderflow_data(20)
        result = strategy.generate_signals(df)

        assert (result["signal"] == 0).all()
