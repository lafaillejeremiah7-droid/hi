"""Tests for scalping features.

Tests session filtering, max trades per day, time-based exits,
and scalping-specific metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics import (
    compute_all_metrics,
    compute_avg_hold_time_minutes,
    compute_max_consecutive,
    compute_scalping_metrics,
    compute_session_breakdown,
    compute_trades_per_day,
)
from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine, BacktestResult, Trade
from src.strategies.order_flow_strategy import OrderFlowStrategy


def make_intraday_data(n_days: int = 5, bars_per_day: int = 7) -> pd.DataFrame:
    """Create synthetic intraday OHLCV data for testing.

    Creates 7 hourly bars per day from 09:30 to 15:30 ET.

    Args:
        n_days: Number of trading days.
        bars_per_day: Bars per day (hourly).

    Returns:
        DataFrame with OHLCV and preprocessed columns.
    """
    np.random.seed(42)
    total_bars = n_days * bars_per_day

    # Create intraday timestamps (hourly bars, 09:30-15:30 ET)
    timestamps = []
    for day_offset in range(n_days):
        base_date = pd.Timestamp("2024-01-02", tz="US/Eastern") + pd.Timedelta(days=day_offset)
        # Skip weekends
        while base_date.weekday() >= 5:
            base_date += pd.Timedelta(days=1)
        for hour_offset in range(bars_per_day):
            ts = base_date.replace(hour=9, minute=30) + pd.Timedelta(hours=hour_offset)
            timestamps.append(ts)

    index = pd.DatetimeIndex(timestamps)
    actual_bars = len(index)

    base = np.cumsum(np.random.randn(actual_bars) * 1.0) + 15000
    open_p = base + np.random.randn(actual_bars) * 0.3
    close_p = base + np.random.randn(actual_bars) * 0.3
    high_p = np.maximum(open_p, close_p) + np.abs(np.random.randn(actual_bars) * 0.5)
    low_p = np.minimum(open_p, close_p) - np.abs(np.random.randn(actual_bars) * 0.5)
    volume = np.random.randint(1000, 10000, actual_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        },
        index=index,
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


class TestSessionFilter:
    """Tests for trading session filtering."""

    def test_session_filter_blocks_outside_hours(self):
        """Engine does not enter trades outside trading session."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(
            strategy, cost_model,
            trading_session_start="10:00",
            trading_session_end="11:00",
        )

        # Create data with some bars inside and outside session
        df = make_intraday_data(n_days=10, bars_per_day=7)
        result = engine.run(df)

        # All trade entries should be within 10:00-11:00
        for trade in result.trades:
            if trade.entry_time is not None and hasattr(trade.entry_time, "hour"):
                entry_minutes = trade.entry_time.hour * 60 + trade.entry_time.minute
                assert 600 <= entry_minutes < 660, (
                    f"Trade entered outside session at {trade.entry_time}"
                )

    def test_no_session_filter_allows_all_hours(self):
        """Without session filter, trades can happen any time."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)

        df = make_intraday_data(n_days=10, bars_per_day=7)
        result = engine.run(df)

        # Should have no restriction
        assert isinstance(result, BacktestResult)

    def test_session_filter_with_full_day(self):
        """Session filter covering full day allows all trades."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()

        # No filter
        engine_no_filter = BacktestEngine(strategy, cost_model)
        # Filter covering full trading hours
        engine_full_day = BacktestEngine(
            strategy, cost_model,
            trading_session_start="09:00",
            trading_session_end="17:00",
        )

        df = make_intraday_data(n_days=10, bars_per_day=7)
        result_no_filter = engine_no_filter.run(df)
        result_full_day = engine_full_day.run(df)

        # Both should produce same trades since all bars are within 09:00-17:00
        assert len(result_no_filter.trades) == len(result_full_day.trades)


class TestMaxTradesPerDay:
    """Tests for max trades per day limit."""

    def test_max_trades_limits_daily_entries(self):
        """Engine respects max_trades_per_day limit."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(
            strategy, cost_model,
            max_trades_per_day=1,
        )

        df = make_intraday_data(n_days=10, bars_per_day=7)
        result = engine.run(df)

        # Count trades per entry day
        from collections import Counter
        daily_counts = Counter()
        for trade in result.trades:
            if trade.entry_time is not None and hasattr(trade.entry_time, "date"):
                daily_counts[trade.entry_time.date()] += 1

        # No day should exceed limit
        for date, count in daily_counts.items():
            assert count <= 1, f"Day {date} had {count} trades, max is 1"

    def test_max_trades_two_per_day(self):
        """Max 2 trades per day is enforced."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(
            strategy, cost_model,
            max_trades_per_day=2,
        )

        df = make_intraday_data(n_days=10, bars_per_day=7)
        result = engine.run(df)

        from collections import Counter
        daily_counts = Counter()
        for trade in result.trades:
            if trade.entry_time is not None and hasattr(trade.entry_time, "date"):
                daily_counts[trade.entry_time.date()] += 1

        for date, count in daily_counts.items():
            assert count <= 2, f"Day {date} had {count} trades, max is 2"

    def test_no_limit_allows_unlimited(self):
        """Without max_trades_per_day, no daily limit is enforced."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model, max_trades_per_day=None)

        df = make_intraday_data(n_days=10, bars_per_day=7)
        result = engine.run(df)
        # Just ensure it runs without error
        assert isinstance(result, BacktestResult)


class TestMaxHoldBars:
    """Tests for dynamic exit management (replaces old max_hold_bars tests).

    max_hold_bars has been removed. Trades exit via:
    - Stop loss (1.0 ATR)
    - Take profit (dynamic: 1.5x or 2.5x ATR based on Z-score)
    - Partial close + trailing stop
    """

    def test_max_hold_bars_ignored(self):
        """max_hold_bars parameter is accepted but ignored."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(
            strategy, cost_model,
            max_hold_bars=2,
        )
        # max_hold_bars is set to None internally
        assert engine.max_hold_bars is None

    def test_trades_exit_via_sl_tp_or_trailing(self):
        """Trades exit via stop_loss, take_profit, or trailing_stop."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)

        df = make_intraday_data(n_days=20, bars_per_day=7)
        result = engine.run(df)

        valid_exit_reasons = {"stop_loss", "take_profit", "trailing_stop", "partial_then_stop"}
        for trade in result.trades:
            assert trade.exit_reason in valid_exit_reasons, (
                f"Unexpected exit_reason: {trade.exit_reason}"
            )

    def test_no_max_hold_exit_reason(self):
        """No trade should have exit_reason='max_hold' anymore."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)

        df = make_intraday_data(n_days=20, bars_per_day=7)
        result = engine.run(df)

        for trade in result.trades:
            assert trade.exit_reason != "max_hold"

    def test_no_max_hold_allows_long_trades(self):
        """Without max_hold_bars, trades can last indefinitely."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model, max_hold_bars=None)

        df = make_intraday_data(n_days=20, bars_per_day=7)
        result = engine.run(df)
        # Just verify it runs
        assert isinstance(result, BacktestResult)


class TestScalpingMetrics:
    """Tests for scalping-specific metrics."""

    def test_max_consecutive_winners(self):
        """Correctly computes max consecutive winners."""
        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1),  # win
            Trade(2, 3, 100, 103, 1, 3, 2, 1),  # win
            Trade(4, 5, 100, 103, 1, 3, 2, 1),  # win
            Trade(6, 7, 100, 97, 1, -3, -4, 1),  # loss
            Trade(8, 9, 100, 102, 1, 2, 1, 1),  # win
        ]
        winners, losers = compute_max_consecutive(trades)
        assert winners == 3
        assert losers == 1

    def test_max_consecutive_losers(self):
        """Correctly computes max consecutive losers."""
        trades = [
            Trade(0, 1, 100, 97, 1, -3, -4, 1),  # loss
            Trade(2, 3, 100, 98, 1, -2, -3, 1),  # loss
            Trade(4, 5, 100, 105, 1, 5, 4, 1),  # win
            Trade(6, 7, 100, 97, 1, -3, -4, 1),  # loss
        ]
        winners, losers = compute_max_consecutive(trades)
        assert winners == 1
        assert losers == 2

    def test_max_consecutive_empty(self):
        """Returns 0,0 for empty trades."""
        winners, losers = compute_max_consecutive([])
        assert winners == 0
        assert losers == 0

    def test_avg_hold_time_with_timestamps(self):
        """Correctly computes average hold time from timestamps."""
        t1_entry = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
        t1_exit = pd.Timestamp("2024-01-02 10:15", tz="US/Eastern")
        t2_entry = pd.Timestamp("2024-01-02 11:00", tz="US/Eastern")
        t2_exit = pd.Timestamp("2024-01-02 11:25", tz="US/Eastern")

        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1, entry_time=t1_entry, exit_time=t1_exit),
            Trade(2, 3, 100, 103, 1, 3, 2, 1, entry_time=t2_entry, exit_time=t2_exit),
        ]
        avg_hold = compute_avg_hold_time_minutes(trades)
        # Expected: (15 + 25) / 2 = 20 minutes
        assert abs(avg_hold - 20.0) < 0.01

    def test_avg_hold_time_no_timestamps(self):
        """Returns 0 when no timestamps available."""
        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1),
            Trade(2, 3, 100, 103, 1, 3, 2, 1),
        ]
        avg_hold = compute_avg_hold_time_minutes(trades)
        assert avg_hold == 0.0

    def test_trades_per_day(self):
        """Correctly computes average trades per day."""
        t1 = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
        t2 = pd.Timestamp("2024-01-02 11:00", tz="US/Eastern")
        t3 = pd.Timestamp("2024-01-03 10:00", tz="US/Eastern")

        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1, entry_time=t1, exit_time=t1),
            Trade(2, 3, 100, 103, 1, 3, 2, 1, entry_time=t2, exit_time=t2),
            Trade(4, 5, 100, 102, 1, 2, 1, 1, entry_time=t3, exit_time=t3),
        ]
        tpd = compute_trades_per_day(trades, total_trading_days=2)
        assert abs(tpd - 1.5) < 0.01

    def test_session_breakdown(self):
        """Correctly splits trades into AM/PM sessions."""
        t_am = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
        t_pm = pd.Timestamp("2024-01-02 14:00", tz="US/Eastern")

        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1, entry_time=t_am, exit_time=t_am),
            Trade(2, 3, 100, 103, 1, 3, 2, 1, entry_time=t_am, exit_time=t_am),
            Trade(4, 5, 100, 97, 1, -3, -4, 1, entry_time=t_pm, exit_time=t_pm),
        ]
        breakdown = compute_session_breakdown(trades)
        assert breakdown["am"]["trades"] == 2
        assert breakdown["am"]["wins"] == 2
        assert breakdown["am"]["win_rate"] == 1.0
        assert breakdown["pm"]["trades"] == 1
        assert breakdown["pm"]["wins"] == 0
        assert breakdown["pm"]["win_rate"] == 0.0

    def test_scalping_metrics_integrated(self):
        """compute_scalping_metrics returns all expected keys."""
        t1 = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
        t2 = pd.Timestamp("2024-01-02 10:20", tz="US/Eastern")

        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1, entry_time=t1, exit_time=t2),
        ]
        metrics = compute_scalping_metrics(trades, total_trading_days=1, point_value=20.0)

        assert "avg_trades_per_day" in metrics
        assert "avg_hold_time_minutes" in metrics
        assert "max_consecutive_winners" in metrics
        assert "max_consecutive_losers" in metrics
        assert "ev_per_trade_dollars" in metrics
        assert "session_breakdown" in metrics
        assert "best_hour" in metrics
        assert "worst_hour" in metrics

    def test_all_metrics_includes_scalping(self):
        """compute_all_metrics includes scalping metrics."""
        equity = pd.Series([0, 4.0], dtype=float)
        t1 = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
        t2 = pd.Timestamp("2024-01-02 10:15", tz="US/Eastern")

        trades = [
            Trade(0, 1, 100, 105, 1, 5, 4, 1, entry_time=t1, exit_time=t2),
        ]
        result = BacktestResult(
            trades=trades,
            equity_gross=equity,
            equity_net=equity,
        )
        metrics = compute_all_metrics(result)

        assert "avg_trades_per_day" in metrics
        assert "avg_hold_time_minutes" in metrics
        assert "max_consecutive_winners" in metrics
        assert "max_consecutive_losers" in metrics
        assert "ev_per_trade_dollars" in metrics


class TestIntradayPreprocessing:
    """Tests for intraday data preprocessing."""

    def test_intraday_cumulative_delta_resets_daily(self):
        """Cumulative delta resets at start of each trading day."""
        from src.data.preprocessor import compute_cumulative_delta, compute_volume_delta

        df = make_intraday_data(n_days=3, bars_per_day=7)
        df["volume_delta"] = compute_volume_delta(df)
        cum_delta = compute_cumulative_delta(df)

        # Check that cumulative delta resets each day
        # The first bar of each day should start fresh
        dates = pd.Series(df.index.date, index=df.index)
        unique_dates = dates.unique()

        for date in unique_dates:
            day_mask = dates == date
            day_deltas = df.loc[day_mask, "volume_delta"]
            day_cum = cum_delta[day_mask]
            # First bar of day should equal that bar's delta
            assert abs(day_cum.iloc[0] - day_deltas.iloc[0]) < 1e-6

    def test_intraday_vwap_resets_daily(self):
        """VWAP resets at start of each trading day."""
        from src.data.preprocessor import compute_vwap

        df = make_intraday_data(n_days=3, bars_per_day=7)
        vwap = compute_vwap(df)

        # VWAP first bar of each day should equal that bar's typical price
        dates = pd.Series(df.index.date, index=df.index)
        unique_dates = dates.unique()

        for date in unique_dates:
            day_mask = dates == date
            day_df = df[day_mask]
            tp = (day_df["high"].iloc[0] + day_df["low"].iloc[0] + day_df["close"].iloc[0]) / 3
            assert abs(vwap[day_mask].iloc[0] - tp) < 1e-6

    def test_preprocess_intraday_data(self):
        """Full preprocess works with intraday data."""
        from src.data.preprocessor import preprocess

        df = make_intraday_data(n_days=5, bars_per_day=7)
        result = preprocess(df)

        # Should have all expected columns
        assert "volume_delta" in result.columns
        assert "cumulative_delta" in result.columns
        assert "relative_volume" in result.columns
        assert "vwap" in result.columns
        assert "support_1" in result.columns
        assert "resistance_1" in result.columns

        # Should also have time-of-day relative volume for intraday
        assert "relative_volume_tod" in result.columns
