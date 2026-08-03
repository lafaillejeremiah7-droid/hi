"""Lookahead regression tests.

Verifies that signals at bar i depend only on data from bars <= i.
Any indicator or strategy that peeks at future data will fail these tests.

Methodology: For each bar i with a non-zero signal, replace all data after
bar i with random noise. If the signal at bar i changes, it depended on
future data (lookahead bias).
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.order_flow import (
    cumulative_delta_divergence,
    detect_absorption,
    detect_failed_auctions,
    detect_stacked_imbalances,
    detect_trapped_traders,
)
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.volume_profile_strategy import VolumeProfileStrategy


def make_test_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create deterministic test data with preprocessed columns.

    Args:
        n: Number of bars.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV and all preprocessed columns.
    """
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    base = np.cumsum(np.random.randn(n) * 0.5 + 0.1) + 100
    open_prices = base + np.random.randn(n) * 0.3
    close_prices = base + np.random.randn(n) * 0.3
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 0.5)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 0.5)
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

    # Add preprocessed columns
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


def corrupt_future_data(df: pd.DataFrame, bar_idx: int, seed: int = 99) -> pd.DataFrame:
    """Replace all data after bar_idx with random noise.

    This ensures that any signal at bar_idx that uses future data
    will produce a different result on the corrupted DataFrame.

    Args:
        df: Original DataFrame.
        bar_idx: The bar index up to which data is preserved.
        seed: Random seed for noise generation.

    Returns:
        New DataFrame with future bars replaced by noise.
    """
    df_copy = df.copy()
    n_future = len(df) - bar_idx - 1

    if n_future <= 0:
        return df_copy

    rng = np.random.default_rng(seed)

    # Replace OHLCV with random values that are clearly different
    future_slice = slice(bar_idx + 1, None)
    noise_base = rng.uniform(500, 600, n_future)  # Very different price level

    df_copy.iloc[bar_idx + 1 :, df_copy.columns.get_loc("open")] = noise_base
    df_copy.iloc[bar_idx + 1 :, df_copy.columns.get_loc("close")] = noise_base + rng.uniform(-5, 5, n_future)
    df_copy.iloc[bar_idx + 1 :, df_copy.columns.get_loc("high")] = noise_base + 10
    df_copy.iloc[bar_idx + 1 :, df_copy.columns.get_loc("low")] = noise_base - 10
    df_copy.iloc[bar_idx + 1 :, df_copy.columns.get_loc("volume")] = rng.uniform(100, 200, n_future)

    # Recompute derived columns from scratch using corrupted raw data
    bar_range = (df_copy["high"] - df_copy["low"]).replace(0, 1)
    df_copy["volume_delta"] = df_copy["volume"] * (df_copy["close"] - df_copy["open"]) / bar_range
    df_copy["cumulative_delta"] = df_copy["volume_delta"].cumsum()
    df_copy["relative_volume"] = df_copy["volume"] / df_copy["volume"].rolling(20, min_periods=1).mean()
    df_copy["resistance_1"] = df_copy["high"].rolling(20, min_periods=1).max()
    df_copy["support_1"] = df_copy["low"].rolling(20, min_periods=1).min()
    df_copy["resistance_2"] = df_copy["resistance_1"] * 0.99
    df_copy["support_2"] = df_copy["support_1"] * 1.01
    dist_sup = (df_copy["close"] - df_copy["support_1"]).abs()
    dist_res = (df_copy["resistance_1"] - df_copy["close"]).abs()
    df_copy["nearest_sr_distance"] = (
        pd.concat([dist_sup, dist_res], axis=1).min(axis=1) / df_copy["close"]
    )
    df_copy["vwap"] = (df_copy["close"] * df_copy["volume"]).cumsum() / df_copy["volume"].cumsum()

    return df_copy


class TestFailedAuctionsNoLookahead:
    """Verify detect_failed_auctions does not use future data."""

    def test_signal_independent_of_future_bars(self):
        """Signals at bar i should not change when future data is corrupted."""
        df = make_test_data(200)
        original_signals = detect_failed_auctions(df)

        # Find bars with non-zero signals
        signal_bars = original_signals[original_signals != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No failed auction signals generated on test data")

        for bar_label in signal_bars[:10]:  # Test up to 10 signal bars
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted_signals = detect_failed_auctions(corrupted_df)

            assert original_signals.iloc[bar_idx] == corrupted_signals.iloc[bar_idx], (
                f"Failed auction signal at bar {bar_idx} changed when future "
                f"data was corrupted. Original={original_signals.iloc[bar_idx]}, "
                f"Corrupted={corrupted_signals.iloc[bar_idx]}. "
                f"This indicates lookahead bias."
            )


class TestTrappedTradersNoLookahead:
    """Verify detect_trapped_traders does not use future data."""

    def test_signal_independent_of_future_bars(self):
        """Signals at bar i should not change when future data is corrupted."""
        df = make_test_data(200)
        original_signals = detect_trapped_traders(df)

        signal_bars = original_signals[original_signals != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No trapped trader signals generated on test data")

        for bar_label in signal_bars[:10]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted_signals = detect_trapped_traders(corrupted_df)

            assert original_signals.iloc[bar_idx] == corrupted_signals.iloc[bar_idx], (
                f"Trapped traders signal at bar {bar_idx} changed when future "
                f"data was corrupted. This indicates lookahead bias."
            )


class TestOrderFlowStrategyNoLookahead:
    """Verify OrderFlowStrategy signals do not use future data."""

    def test_signals_independent_of_future_bars(self):
        """Strategy signals at bar i depend only on data <= bar i."""
        df = make_test_data(200)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        original_result = strategy.generate_signals(df)
        original_signals = original_result["signal"]

        signal_bars = original_signals[original_signals != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No Order Flow signals generated on test data")

        for bar_label in signal_bars[:10]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted_result = strategy.generate_signals(corrupted_df)
            corrupted_signals = corrupted_result["signal"]

            assert original_signals.iloc[bar_idx] == corrupted_signals.iloc[bar_idx], (
                f"Order Flow signal at bar {bar_idx} changed when future "
                f"data was corrupted. Original={original_signals.iloc[bar_idx]}, "
                f"Corrupted={corrupted_signals.iloc[bar_idx]}. "
                f"This indicates lookahead bias."
            )


class TestVolumeProfileStrategyNoLookahead:
    """Verify VolumeProfileStrategy signals do not use future data."""

    def test_signals_independent_of_future_bars(self):
        """Strategy signals at bar i depend only on data <= bar i."""
        df = make_test_data(300)
        strategy = VolumeProfileStrategy()
        original_result = strategy.generate_signals(df)
        original_signals = original_result["signal"]

        signal_bars = original_signals[original_signals != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No Volume Profile signals generated on test data")

        for bar_label in signal_bars[:10]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted_result = strategy.generate_signals(corrupted_df)
            corrupted_signals = corrupted_result["signal"]

            assert original_signals.iloc[bar_idx] == corrupted_signals.iloc[bar_idx], (
                f"Volume Profile signal at bar {bar_idx} changed when future "
                f"data was corrupted. Original={original_signals.iloc[bar_idx]}, "
                f"Corrupted={corrupted_signals.iloc[bar_idx]}. "
                f"This indicates lookahead bias."
            )


class TestTakeProfitNoLookahead:
    """Verify take_profit calculations do not use future data."""

    def test_volume_profile_take_profit_no_lookahead(self):
        """VolumeProfileStrategy.get_take_profit uses only backward data."""
        df = make_test_data(200)
        strategy = VolumeProfileStrategy()

        # Test several positions in the middle of the dataset
        for idx in [50, 75, 100, 125, 150]:
            for direction in [1, -1]:
                # Get take profit with full data
                tp_full = strategy.get_take_profit(df, idx, direction)

                # Get take profit with future data corrupted
                corrupted_df = corrupt_future_data(df, idx)
                tp_corrupted = strategy.get_take_profit(corrupted_df, idx, direction)

                assert tp_full == tp_corrupted, (
                    f"VolumeProfileStrategy.get_take_profit at idx={idx}, "
                    f"direction={direction} changed when future data was "
                    f"corrupted ({tp_full} vs {tp_corrupted}). "
                    f"This indicates lookahead bias."
                )

    def test_order_flow_take_profit_no_lookahead(self):
        """OrderFlowStrategy.get_take_profit uses only backward data (ATR-based)."""
        df = make_test_data(200)
        strategy = OrderFlowStrategy()

        for idx in [50, 75, 100, 125, 150]:
            for direction in [1, -1]:
                tp_full = strategy.get_take_profit(df, idx, direction)
                corrupted_df = corrupt_future_data(df, idx)
                tp_corrupted = strategy.get_take_profit(corrupted_df, idx, direction)

                assert tp_full == tp_corrupted, (
                    f"OrderFlowStrategy.get_take_profit at idx={idx}, "
                    f"direction={direction} changed when future data was "
                    f"corrupted ({tp_full} vs {tp_corrupted}). "
                    f"This indicates lookahead bias."
                )


class TestIndicatorsNoLookahead:
    """Verify all individual indicators are free of lookahead bias."""

    def test_absorption_no_lookahead(self):
        """detect_absorption uses only current and past data."""
        df = make_test_data(200)
        original = detect_absorption(df)

        signal_bars = original[original != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No absorption signals generated")

        for bar_label in signal_bars[:10]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted = detect_absorption(corrupted_df)
            assert original.iloc[bar_idx] == corrupted.iloc[bar_idx], (
                f"Absorption signal at bar {bar_idx} depends on future data."
            )

    def test_delta_divergence_no_lookahead(self):
        """cumulative_delta_divergence uses only current and past data."""
        df = make_test_data(200)
        original = cumulative_delta_divergence(df)

        signal_bars = original[original != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No divergence signals generated")

        for bar_label in signal_bars[:10]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted = cumulative_delta_divergence(corrupted_df)
            assert original.iloc[bar_idx] == corrupted.iloc[bar_idx], (
                f"Delta divergence signal at bar {bar_idx} depends on future data."
            )

    def test_stacked_imbalances_no_lookahead(self):
        """detect_stacked_imbalances uses only current and past data."""
        df = make_test_data(200)
        original = detect_stacked_imbalances(df)

        signal_bars = original[original != 0].index
        if len(signal_bars) == 0:
            pytest.skip("No stacked imbalance signals generated")

        for bar_label in signal_bars[:10]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted_df = corrupt_future_data(df, bar_idx)
            corrupted = detect_stacked_imbalances(corrupted_df)
            assert original.iloc[bar_idx] == corrupted.iloc[bar_idx], (
                f"Stacked imbalance signal at bar {bar_idx} depends on future data."
            )
