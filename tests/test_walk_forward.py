"""Tests for the walk-forward analysis module.

Tests walk-forward splits are non-overlapping, OOS windows are truly
out-of-sample (no future data leakage), consistency ratio calculation,
edge cases with minimal data, and that train windows never include OOS data.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.walk_forward import WalkForwardAnalyzer, WalkForwardResults
from src.backtester.costs import CostModel
from src.strategies.order_flow_strategy import OrderFlowStrategy


def make_walk_forward_data(months: int = 24) -> pd.DataFrame:
    """Create synthetic intraday-like data spanning multiple months.

    Uses business-day hourly frequency to simulate realistic market data.

    Args:
        months: Number of months of data to generate.

    Returns:
        DataFrame with hourly OHLCV data and indicator columns.
    """
    np.random.seed(42)
    # Generate enough calendar time to span the desired months
    # Use daily data at business-day frequency, then multiply by hours per day
    start_date = "2023-01-01"
    end_date = pd.Timestamp(start_date) + pd.DateOffset(months=months)

    # Generate business-day dates spanning the full period
    bdays = pd.bdate_range(start=start_date, end=end_date)

    # For each business day, generate 7 hourly bars (9:00-15:00)
    all_timestamps = []
    for day in bdays:
        for hour in range(9, 16):
            all_timestamps.append(day + pd.Timedelta(hours=hour))

    dates = pd.DatetimeIndex(all_timestamps)
    n = len(dates)

    if n == 0:
        # Fallback for very short periods
        dates = pd.date_range(start_date, periods=100, freq="h")
        n = len(dates)

    base = np.cumsum(np.random.randn(n) * 2.0) + 15000
    open_p = base + np.random.randn(n) * 1.0
    close_p = base + np.random.randn(n) * 1.0
    high_p = np.maximum(open_p, close_p) + np.abs(np.random.randn(n) * 2.0)
    low_p = np.minimum(open_p, close_p) - np.abs(np.random.randn(n) * 2.0)
    volume = np.random.randint(1000, 10000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        },
        index=dates,
    )

    # Add preprocessed columns needed by strategies
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


class TestWalkForwardWindowGeneration:
    """Tests for walk-forward window generation."""

    def test_windows_are_non_overlapping(self):
        """Walk-forward OOS windows do not overlap with each other."""
        df = make_walk_forward_data(18)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=6,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        windows = analyzer._generate_windows(df)
        assert len(windows) > 0

        # Check that test windows don't overlap
        for i in range(1, len(windows)):
            prev_train, prev_test = windows[i - 1]
            curr_train, curr_test = windows[i]

            # Previous test end should be <= current test start
            assert prev_test.index[-1] <= curr_test.index[0], (
                f"Test window {i-1} overlaps with test window {i}: "
                f"{prev_test.index[-1]} > {curr_test.index[0]}"
            )

    def test_oos_windows_are_truly_out_of_sample(self):
        """OOS test data never appears in the corresponding train data."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        windows = analyzer._generate_windows(df)
        assert len(windows) > 0

        for i, (train_df, test_df) in enumerate(windows):
            # No test data timestamp should appear in train data
            train_set = set(train_df.index)
            test_set = set(test_df.index)
            overlap = train_set & test_set
            assert len(overlap) == 0, (
                f"Window {i}: {len(overlap)} bars appear in both train and test"
            )

    def test_train_windows_do_not_include_future_data(self):
        """Train windows only contain data that precedes the test window."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        windows = analyzer._generate_windows(df)
        assert len(windows) > 0

        for i, (train_df, test_df) in enumerate(windows):
            # All train data must be earlier than the test data start
            assert train_df.index[-1] < test_df.index[0], (
                f"Window {i}: train end {train_df.index[-1]} >= test start {test_df.index[0]}"
            )

    def test_anchored_windows_start_from_beginning(self):
        """Anchored mode: all train windows start from the same point."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=True,
        )

        windows = analyzer._generate_windows(df)
        assert len(windows) > 1

        # All train windows should start at or near the same point
        first_train_start = windows[0][0].index[0]
        for i, (train_df, _) in enumerate(windows):
            assert train_df.index[0] == first_train_start, (
                f"Window {i}: anchored train start {train_df.index[0]} != {first_train_start}"
            )

    def test_rolling_windows_advance(self):
        """Rolling mode: train windows advance forward over time."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        windows = analyzer._generate_windows(df)
        assert len(windows) > 2

        # Later train windows should start later than earlier ones
        for i in range(1, len(windows)):
            prev_start = windows[i - 1][0].index[0]
            curr_start = windows[i][0].index[0]
            assert curr_start >= prev_start, (
                f"Window {i}: rolling train start should advance, "
                f"got {curr_start} <= {prev_start}"
            )


class TestWalkForwardExecution:
    """Tests for walk-forward analysis execution."""

    def test_run_returns_results(self):
        """Walk-forward run returns valid WalkForwardResults."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)

        assert isinstance(results, WalkForwardResults)
        assert results.total_oos_windows > 0
        assert len(results.windows) == results.total_oos_windows
        assert len(results.per_window_returns) == results.total_oos_windows

    def test_consistency_ratio_calculation(self):
        """Consistency ratio is correctly calculated."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)

        # Manually verify consistency ratio
        profitable = sum(1 for r in results.per_window_returns if r > 0)
        expected_ratio = profitable / len(results.per_window_returns) if results.per_window_returns else 0.0
        assert abs(results.consistency_ratio - expected_ratio) < 1e-10
        assert results.profitable_windows == profitable

    def test_consistency_ratio_bounds(self):
        """Consistency ratio is between 0 and 1."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)
        assert 0.0 <= results.consistency_ratio <= 1.0

    def test_minimal_data_returns_empty(self):
        """Walk-forward with insufficient data returns empty results."""
        # Only 2 months of data -- not enough for a 6-month train + 1-month test
        df = make_walk_forward_data(2)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=6,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)
        assert results.total_oos_windows == 0
        assert results.consistency_ratio == 0.0

    def test_combined_oos_equity_is_stitched(self):
        """Combined OOS equity curve is properly stitched from all windows."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)

        if results.total_oos_windows > 0:
            # Combined equity should have data
            assert len(results.combined_oos_equity) > 0
            # First value should be the first window's first equity value
            # (which is 0 since equity starts from 0)

    def test_from_config_creates_analyzer(self):
        """from_config class method creates a valid analyzer."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        config = {
            "walk_forward": {
                "enabled": True,
                "train_window_months": 6,
                "test_window_months": 1,
                "step_months": 1,
                "anchored": False,
            },
            "costs": {
                "slippage_points": 0.75,
                "commission_per_round_trip": 4.50,
                "point_value": 20.0,
            },
        }

        analyzer = WalkForwardAnalyzer.from_config(
            strategy=strategy,
            config=config,
        )

        assert analyzer.train_window_months == 6
        assert analyzer.test_window_months == 1
        assert analyzer.step_months == 1
        assert analyzer.anchored is False
        assert analyzer.point_value == 20.0


class TestWalkForwardMetrics:
    """Tests for walk-forward derived metrics."""

    def test_degradation_metric_finite(self):
        """Degradation metric is a finite number."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)
        assert np.isfinite(results.degradation_metric)

    def test_walk_forward_efficiency_finite(self):
        """Walk-forward efficiency is a finite number."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)
        assert np.isfinite(results.walk_forward_efficiency)

    def test_regime_warning_is_boolean(self):
        """Regime warning flag is a boolean."""
        df = make_walk_forward_data(12)
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        analyzer = WalkForwardAnalyzer(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=4,
            test_window_months=1,
            step_months=1,
            anchored=False,
        )

        results = analyzer.run(df)
        assert isinstance(results.regime_warning, bool)
