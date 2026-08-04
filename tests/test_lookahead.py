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
from src.strategies.simple_strategy import (
    SimpleStrategy,
    heavy_volume_node,
    rolling_delta_zscore,
)
from src.indicators.williams import (
    greatest_swing_values,
    gsv_triggers,
    oops_triggers,
    smash_day_triggers,
    tdom_bias_flags,
    tdom_bias_table,
    trading_day_of_month,
)
from src.strategies.volume_profile_strategy import VolumeProfileStrategy
from src.strategies.williams_strategy import WilliamsStrategy


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


def make_orderflow_test_data(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """Create 5-min data with a real order flow delta column.

    Args:
        n: Number of bars.
        seed: Random seed.

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

    return pd.DataFrame(
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


class TestSimpleStrategyNoLookahead:
    """Verify SimpleStrategy signals depend only on data at or before bar i."""

    def test_signals_unchanged_when_future_deleted(self):
        """Signal at bar i is identical when every bar after i is deleted."""
        df = make_orderflow_test_data(400)
        strategy = SimpleStrategy()
        full_signals = strategy.generate_signals(df)["signal"]

        signal_bars = full_signals[full_signals != 0].index
        assert len(signal_bars) > 0, "Fixture produced no signals to test"

        for bar_label in signal_bars[:15]:
            bar_idx = df.index.get_loc(bar_label)
            truncated = df.iloc[: bar_idx + 1].copy()
            truncated_signals = strategy.generate_signals(truncated)["signal"]

            assert full_signals.iloc[bar_idx] == truncated_signals.iloc[bar_idx], (
                f"SimpleStrategy signal at bar {bar_idx} changed when all data "
                f"after bar {bar_idx} was deleted. This indicates lookahead bias."
            )

    def test_flat_bars_also_unchanged_when_future_deleted(self):
        """Bars with no signal must also stay signal-free after truncation."""
        df = make_orderflow_test_data(400)
        strategy = SimpleStrategy()
        full_signals = strategy.generate_signals(df)["signal"]

        for bar_idx in range(200, 260):
            truncated = df.iloc[: bar_idx + 1].copy()
            truncated_signals = strategy.generate_signals(truncated)["signal"]
            assert full_signals.iloc[bar_idx] == truncated_signals.iloc[bar_idx], (
                f"SimpleStrategy signal at bar {bar_idx} changed after truncation."
            )

    def test_heavy_volume_node_unchanged_when_future_deleted(self):
        """The volume profile level at bar i uses bars <= i only."""
        df = make_orderflow_test_data(400)
        full_node = heavy_volume_node(df, lookback=78)

        for bar_idx in [100, 150, 200, 275, 399]:
            truncated_node = heavy_volume_node(df.iloc[: bar_idx + 1], lookback=78)
            assert full_node.iloc[bar_idx] == truncated_node.iloc[bar_idx], (
                f"Heavy volume node at bar {bar_idx} depends on future data."
            )

    def test_delta_zscore_unchanged_when_future_deleted(self):
        """The delta Z-score at bar i uses bars <= i only."""
        df = make_orderflow_test_data(400)
        full_z = rolling_delta_zscore(df)

        for bar_idx in [100, 150, 200, 275, 399]:
            truncated_z = rolling_delta_zscore(df.iloc[: bar_idx + 1])
            assert full_z.iloc[bar_idx] == pytest.approx(truncated_z.iloc[bar_idx]), (
                f"Delta Z-score at bar {bar_idx} depends on future data."
            )

    def test_signals_unchanged_when_future_corrupted(self):
        """Signal at bar i survives replacing future bars with noise."""
        df = make_orderflow_test_data(400)
        strategy = SimpleStrategy()
        full_signals = strategy.generate_signals(df)["signal"]

        signal_bars = full_signals[full_signals != 0].index
        assert len(signal_bars) > 0, "Fixture produced no signals to test"

        rng = np.random.default_rng(99)
        for bar_label in signal_bars[:15]:
            bar_idx = df.index.get_loc(bar_label)
            corrupted = df.copy()
            n_future = len(df) - bar_idx - 1
            if n_future <= 0:
                continue
            noise = rng.uniform(5000, 6000, n_future)
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("open")] = noise
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("close")] = noise
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("high")] = noise + 10
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("low")] = noise - 10
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("volume")] = 9e6
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("delta")] = 9e6

            corrupted_signals = strategy.generate_signals(corrupted)["signal"]
            assert full_signals.iloc[bar_idx] == corrupted_signals.iloc[bar_idx], (
                f"SimpleStrategy signal at bar {bar_idx} changed when future data "
                f"was corrupted. This indicates lookahead bias."
            )


def make_daily_test_data(n: int = 400, seed: int = 23) -> pd.DataFrame:
    """Create daily RTH bars for the Williams look-ahead tests.

    Args:
        n: Number of trading days.
        seed: Random seed.

    Returns:
        DataFrame with the columns the Williams components read.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=n, freq="B", tz="US/Eastern") + pd.Timedelta(
        hours=9, minutes=30
    )

    base = np.cumsum(rng.normal(0, 70, n)) + 14000
    open_p = base + rng.normal(0, 25, n)
    close_p = base + rng.normal(0, 25, n)
    high_p = np.maximum(open_p, close_p) + np.abs(rng.normal(0, 50, n))
    low_p = np.minimum(open_p, close_p) - np.abs(rng.normal(0, 50, n))

    df = pd.DataFrame(
        {"open": open_p, "high": high_p, "low": low_p, "close": close_p},
        index=index,
    )
    df.index.name = "Date"
    df["volume"] = rng.uniform(1e5, 1e6, n)
    df["n_bars"] = 390
    df["prior_close"] = df["close"].shift(1)
    high_first = rng.random(n) < 0.5
    df["high_time"] = np.where(high_first, index + pd.Timedelta(hours=1),
                               index + pd.Timedelta(hours=3))
    df["low_time"] = np.where(high_first, index + pd.Timedelta(hours=3),
                              index + pd.Timedelta(hours=1))
    return df


# Bars checked by the Williams truncation tests: a mix of positions well past
# the indicator warm-up.
WILLIAMS_TRUNCATION_BARS = [60, 97, 155, 201, 280, 333, 399]


class TestWilliamsIndicatorsNoLookahead:
    """Every Williams component at bar i must use only bars <= i."""

    def test_gsv_averages_unchanged_when_future_deleted(self):
        """GSV averages at bar i survive deleting every bar after i."""
        df = make_daily_test_data(400)
        for lookback in (3, 5, 10):
            full = greatest_swing_values(df, lookback)
            for bar_idx in WILLIAMS_TRUNCATION_BARS:
                truncated = greatest_swing_values(df.iloc[: bar_idx + 1], lookback)
                for column in ("gsv_buy", "gsv_sell"):
                    expected = full[column].iloc[bar_idx]
                    actual = truncated[column].iloc[bar_idx]
                    assert (pd.isna(expected) and pd.isna(actual)) or expected == pytest.approx(
                        actual
                    ), (
                        f"{column} (N={lookback}) at bar {bar_idx} changed when all data "
                        f"after bar {bar_idx} was deleted. This indicates lookahead bias."
                    )

    def test_gsv_triggers_unchanged_when_future_deleted(self):
        """GSV entry stop levels at bar i survive truncation at i."""
        df = make_daily_test_data(400)
        full = gsv_triggers(df, lookback=5, multiplier=0.8)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            truncated = gsv_triggers(df.iloc[: bar_idx + 1], lookback=5, multiplier=0.8)
            for column in ("long_trigger", "short_trigger"):
                expected = full[column].iloc[bar_idx]
                actual = truncated[column].iloc[bar_idx]
                assert (pd.isna(expected) and pd.isna(actual)) or expected == pytest.approx(
                    actual
                ), f"GSV {column} at bar {bar_idx} depends on future data."

    def test_oops_triggers_unchanged_when_future_deleted(self):
        """Oops! levels at bar i survive truncation at i."""
        df = make_daily_test_data(400)
        full = oops_triggers(df)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            truncated = oops_triggers(df.iloc[: bar_idx + 1])
            for column in ("long_trigger", "short_trigger"):
                expected = full[column].iloc[bar_idx]
                actual = truncated[column].iloc[bar_idx]
                assert (pd.isna(expected) and pd.isna(actual)) or expected == pytest.approx(
                    actual
                ), f"Oops {column} at bar {bar_idx} depends on future data."

    def test_smash_day_triggers_unchanged_when_future_deleted(self):
        """Smash Day levels at bar i survive truncation at i."""
        df = make_daily_test_data(400)
        full = smash_day_triggers(df, lookback=5)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            truncated = smash_day_triggers(df.iloc[: bar_idx + 1], lookback=5)
            for column in ("long_trigger", "short_trigger"):
                expected = full[column].iloc[bar_idx]
                actual = truncated[column].iloc[bar_idx]
                assert (pd.isna(expected) and pd.isna(actual)) or expected == pytest.approx(
                    actual
                ), f"Smash Day {column} at bar {bar_idx} depends on future data."

    def test_trading_day_of_month_unchanged_when_future_deleted(self):
        """The trading-day-of-month index at bar i survives truncation at i."""
        df = make_daily_test_data(400)
        full = trading_day_of_month(df.index)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            truncated = trading_day_of_month(df.index[: bar_idx + 1])
            assert full.iloc[bar_idx] == truncated.iloc[bar_idx], (
                f"TDOM index at bar {bar_idx} depends on future data."
            )

    def test_tdom_table_reads_only_the_rows_it_is_given(self):
        """A TDOM table fitted on bars <= i is unaffected by bars after i."""
        df = make_daily_test_data(400)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            from_prefix = tdom_bias_table(df.iloc[: bar_idx + 1], min_observations=3)
            corrupted = df.copy()
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("close")] += 5000
            corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc("open")] -= 5000
            from_corrupted = tdom_bias_table(
                corrupted.iloc[: bar_idx + 1], min_observations=3
            )
            assert from_prefix == from_corrupted, (
                f"TDOM table fitted through bar {bar_idx} changed when later bars were "
                f"corrupted. This indicates lookahead bias."
            )

    def test_tdom_flags_unchanged_when_future_deleted(self):
        """Bias flags at bar i depend only on the frozen table and the calendar."""
        df = make_daily_test_data(400)
        table = tdom_bias_table(df.iloc[:200], min_observations=3)
        full = tdom_bias_flags(df.index, table)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            truncated = tdom_bias_flags(df.index[: bar_idx + 1], table)
            assert full.iloc[bar_idx] == truncated.iloc[bar_idx], (
                f"TDOM bias flag at bar {bar_idx} depends on future data."
            )


class TestWilliamsStrategyNoLookahead:
    """WilliamsStrategy signals at bar i must be identical after truncation at i."""

    PARAM_SETS = [
        {"components": ("gsv",), "gsv_lookback": 10, "gsv_multiplier": 1.0},
        {"components": ("gsv", "oops"), "gsv_lookback": 5, "gsv_multiplier": 0.6},
        {"components": ("gsv", "oops", "smash"), "gsv_lookback": 3,
         "gsv_multiplier": 0.8, "smash_lookback": 5},
    ]

    def _assert_bar_matches(self, full, truncated, bar_idx, label):
        """Assert every signal column agrees at one bar."""
        for column in ("signal", "long_trigger", "short_trigger"):
            expected = full[column].iloc[bar_idx]
            actual = truncated[column].iloc[bar_idx]
            if isinstance(expected, float) and pd.isna(expected):
                assert pd.isna(actual), (
                    f"{label}: {column} at bar {bar_idx} became {actual} when all data "
                    f"after bar {bar_idx} was deleted. This indicates lookahead bias."
                )
            else:
                assert expected == pytest.approx(actual), (
                    f"{label}: {column} at bar {bar_idx} changed from {expected} to "
                    f"{actual} when all data after bar {bar_idx} was deleted. "
                    f"This indicates lookahead bias."
                )

    def test_signals_unchanged_when_future_deleted(self):
        """The core regression: delete everything after bar i, signals at i are identical."""
        df = make_daily_test_data(400)
        table = tdom_bias_table(df.iloc[:200], min_observations=3)

        for params in self.PARAM_SETS:
            strategy = WilliamsStrategy(params={**params, "tdom_bias": table})
            full = strategy.generate_signals(df)
            label = "+".join(params["components"])

            for bar_idx in WILLIAMS_TRUNCATION_BARS:
                truncated = strategy.generate_signals(df.iloc[: bar_idx + 1])
                self._assert_bar_matches(full, truncated, bar_idx, label)

    def test_signals_unchanged_with_the_tdom_filter(self):
        """The TDOM filter uses a frozen table, so truncation cannot move a signal."""
        df = make_daily_test_data(400)
        table = tdom_bias_table(df.iloc[:200], min_observations=3)
        strategy = WilliamsStrategy(
            params={"components": ("gsv",), "tdom_bias": table, "tdom_filter": True}
        )
        full = strategy.generate_signals(df)

        for bar_idx in WILLIAMS_TRUNCATION_BARS:
            truncated = strategy.generate_signals(df.iloc[: bar_idx + 1])
            self._assert_bar_matches(full, truncated, bar_idx, "gsv+tdom filter")

    def test_armed_signals_exist_so_the_test_is_meaningful(self):
        """The fixture must actually arm triggers, or the test proves nothing."""
        df = make_daily_test_data(400)
        signals = WilliamsStrategy(params={"components": ("gsv",)}).generate_signals(df)

        armed = signals.loc[df.index[WILLIAMS_TRUNCATION_BARS], "long_trigger"].notna()
        assert armed.any(), "Fixture produced no armed triggers at the tested bars"

    def test_signals_unchanged_when_future_corrupted(self):
        """Replacing future bars with noise cannot change a signal at bar i."""
        df = make_daily_test_data(400)
        strategy = WilliamsStrategy(
            params={"components": ("gsv", "oops", "smash"), "gsv_lookback": 5}
        )
        full = strategy.generate_signals(df)

        rng = np.random.default_rng(5)
        for bar_idx in WILLIAMS_TRUNCATION_BARS[:-1]:
            corrupted = df.copy()
            n_future = len(df) - bar_idx - 1
            noise = rng.uniform(40000, 50000, n_future)
            for column, offset in (("open", 0.0), ("close", 5.0), ("high", 200.0),
                                   ("low", -200.0)):
                corrupted.iloc[bar_idx + 1 :, corrupted.columns.get_loc(column)] = (
                    noise + offset
                )
            corrupted["prior_close"] = corrupted["close"].shift(1)

            corrupted_signals = strategy.generate_signals(corrupted)
            self._assert_bar_matches(full, corrupted_signals, bar_idx, "gsv+oops+smash")
