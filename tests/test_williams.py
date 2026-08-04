"""Tests for the Williams daily-bar system.

Covers the daily bar construction, each of the four components, the embargoed
split, and the daily engine's fill logic - in particular that a position gapped
through its stop is filled at the open and not at the stop.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester.costs import CostModel
from src.backtester.daily_engine import (
    DailyBacktestEngine,
    daily_pnl_units,
    gap_fill_statistics,
    replay_prop_account,
)
from src.backtester.embargo_split import build_embargo_split, segment_by_name
from src.data.daily_bars import DAILY_COLUMNS, resample_to_daily_rth
from src.indicators.williams import (
    greatest_swing_values,
    gsv_triggers,
    oops_triggers,
    smash_day_triggers,
    tdom_bias_flags,
    tdom_bias_table,
    trading_day_of_month,
)
from src.strategies.williams_strategy import WilliamsStrategy

ZERO_COSTS = CostModel(
    base_slippage_points=0.0, commission_per_round_trip=0.0, point_value=20.0
)


def make_daily(rows: list[dict], start: str = "2024-01-02") -> pd.DataFrame:
    """Build a daily RTH frame from explicit OHLC rows.

    Args:
        rows: Dicts with open/high/low/close keys.
        start: First session date.

    Returns:
        Daily frame with every column the engine expects.
    """
    index = pd.date_range(start, periods=len(rows), freq="B", tz="US/Eastern") + pd.Timedelta(
        hours=9, minutes=30
    )
    frame = pd.DataFrame(rows, index=index)
    frame.index.name = "Date"
    frame["volume"] = 1000.0
    frame["n_bars"] = 390
    frame["prior_close"] = frame["close"].shift(1)
    # High first, then low, unless a row says otherwise.
    frame["high_time"] = index + pd.Timedelta(hours=1)
    frame["low_time"] = index + pd.Timedelta(hours=2)
    return frame


def make_random_daily(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Build a random-walk daily frame for property tests."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=n, freq="B", tz="US/Eastern") + pd.Timedelta(
        hours=9, minutes=30
    )

    base = np.cumsum(rng.normal(0, 60, n)) + 15000
    open_p = base + rng.normal(0, 20, n)
    close_p = base + rng.normal(0, 20, n)
    high_p = np.maximum(open_p, close_p) + np.abs(rng.normal(0, 40, n))
    low_p = np.minimum(open_p, close_p) - np.abs(rng.normal(0, 40, n))

    frame = pd.DataFrame(
        {"open": open_p, "high": high_p, "low": low_p, "close": close_p},
        index=index,
    )
    frame.index.name = "Date"
    frame["volume"] = rng.uniform(1e5, 1e6, n)
    frame["n_bars"] = 390
    frame["prior_close"] = frame["close"].shift(1)
    high_first = rng.random(n) < 0.5
    frame["high_time"] = np.where(high_first, index + pd.Timedelta(hours=1),
                                  index + pd.Timedelta(hours=3))
    frame["low_time"] = np.where(high_first, index + pd.Timedelta(hours=3),
                                 index + pd.Timedelta(hours=1))
    return frame


class TestDailyBarConstruction:
    """Daily RTH resampling from 1-minute bars."""

    def _minute_frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-02 00:00", periods=60 * 24 * 2, freq="1min", tz="UTC")
        rng = np.random.default_rng(3)
        base = np.cumsum(rng.normal(0, 1, len(index))) + 16000
        return pd.DataFrame(
            {
                "Open": base,
                "High": base + 2,
                "Low": base - 2,
                "Close": base + 0.5,
                "Volume": np.full(len(index), 10.0),
            },
            index=index,
        )

    def test_produces_one_bar_per_session_with_expected_columns(self):
        """Resampling yields one bar per day with all documented columns."""
        daily = resample_to_daily_rth(self._minute_frame(), min_session_bars=10)

        assert list(daily.columns) == DAILY_COLUMNS
        assert len(daily) == 2
        assert (daily.index.hour == 9).all()
        assert (daily.index.minute == 30).all()
        assert str(daily.index.tz) == "US/Eastern"

    def test_session_bounds_and_prior_close(self):
        """High/low bound open and close, and prior_close is the previous close."""
        daily = resample_to_daily_rth(self._minute_frame(), min_session_bars=10)

        assert (daily["high"] >= daily[["open", "close"]].max(axis=1)).all()
        assert (daily["low"] <= daily[["open", "close"]].min(axis=1)).all()
        assert pd.isna(daily["prior_close"].iloc[0])
        assert daily["prior_close"].iloc[1] == daily["close"].iloc[0]

    def test_short_sessions_are_dropped(self):
        """A day with too few 1-minute bars is treated as a data gap."""
        minute = self._minute_frame()
        # Keep only 5 session minutes of the second day.
        second_day = minute.index.normalize() == minute.index.normalize()[-1]
        trimmed = pd.concat([minute.loc[~second_day], minute.loc[second_day].iloc[:5]])

        daily = resample_to_daily_rth(trimmed, min_session_bars=120)
        assert len(daily) == 1


class TestGreatestSwingValue:
    """GSV averages and triggers."""

    def test_buy_swing_averages_up_close_days_only(self):
        """gsv_buy averages open-low on days that closed above the prior close."""
        rows = [
            {"open": 100, "high": 110, "low": 90, "close": 105},   # no prior close
            {"open": 105, "high": 115, "low": 95, "close": 110},   # up close, open-low = 10
            {"open": 110, "high": 118, "low": 96, "close": 100},   # down close
            {"open": 100, "high": 112, "low": 80, "close": 108},   # up close, open-low = 20
            {"open": 108, "high": 120, "low": 100, "close": 118},
        ]
        gsv = greatest_swing_values(make_daily(rows), lookback=2)

        # Day 4 (positional 4) is the first day where two up-close days precede it.
        assert gsv["gsv_buy"].iloc[4] == pytest.approx(15.0)
        # Nothing is available before two qualifying days have completed.
        assert gsv["gsv_buy"].iloc[:4].isna().all()

    def test_sell_swing_averages_down_close_days_only(self):
        """gsv_sell averages high-open on days that closed below the prior close."""
        rows = [
            {"open": 100, "high": 110, "low": 90, "close": 105},
            {"open": 105, "high": 125, "low": 95, "close": 100},   # down close, high-open = 20
            {"open": 100, "high": 130, "low": 96, "close": 90},    # down close, high-open = 30
            {"open": 90, "high": 100, "low": 80, "close": 95},
        ]
        gsv = greatest_swing_values(make_daily(rows), lookback=2)
        assert gsv["gsv_sell"].iloc[3] == pytest.approx(25.0)

    def test_triggers_straddle_the_open(self):
        """The buy stop sits above the open and the sell stop below it."""
        df = make_random_daily(120)
        triggers = gsv_triggers(df, lookback=5, multiplier=0.8)

        armed_long = triggers["long_trigger"].notna()
        armed_short = triggers["short_trigger"].notna()
        assert (triggers["long_trigger"][armed_long] > df["open"][armed_long]).all()
        assert (triggers["short_trigger"][armed_short] < df["open"][armed_short]).all()

    def test_gsv_uses_no_data_from_its_own_day(self):
        """Changing a day's own high/low/close cannot change its GSV value."""
        df = make_random_daily(150)
        original = greatest_swing_values(df, lookback=5)

        tampered = df.copy()
        tampered.iloc[100, tampered.columns.get_loc("high")] += 500
        tampered.iloc[100, tampered.columns.get_loc("low")] -= 500
        tampered.iloc[100, tampered.columns.get_loc("close")] += 500
        after = greatest_swing_values(tampered, lookback=5)

        assert original["gsv_buy"].iloc[100] == pytest.approx(after["gsv_buy"].iloc[100])
        assert original["gsv_sell"].iloc[100] == pytest.approx(after["gsv_sell"].iloc[100])


class TestOops:
    """The Oops! pattern."""

    def test_long_arms_only_when_open_is_below_yesterdays_low(self):
        """A buy stop at yesterday's low appears only on a gap-down open."""
        rows = [
            {"open": 100, "high": 110, "low": 95, "close": 105},
            {"open": 90, "high": 108, "low": 88, "close": 100},   # opens below 95
            {"open": 101, "high": 105, "low": 99, "close": 103},   # opens inside
        ]
        triggers = oops_triggers(make_daily(rows))

        assert triggers["long_trigger"].iloc[1] == pytest.approx(95.0)
        assert pd.isna(triggers["long_trigger"].iloc[2])
        assert triggers["short_trigger"].isna().all()

    def test_short_arms_only_when_open_is_above_yesterdays_high(self):
        """A sell stop at yesterday's high appears only on a gap-up open."""
        rows = [
            {"open": 100, "high": 110, "low": 95, "close": 105},
            {"open": 120, "high": 125, "low": 112, "close": 118},  # opens above 110
        ]
        triggers = oops_triggers(make_daily(rows))
        assert triggers["short_trigger"].iloc[1] == pytest.approx(110.0)
        assert triggers["long_trigger"].isna().all()


class TestSmashDay:
    """The Smash Day reversal."""

    def test_long_arms_after_a_close_below_the_prior_range(self):
        """Yesterday closing under the prior N-day low arms a buy stop at its high."""
        rows = [
            {"open": 100, "high": 105, "low": 98, "close": 102},
            {"open": 102, "high": 106, "low": 99, "close": 104},
            {"open": 104, "high": 107, "low": 97, "close": 96},   # closes below the 2-day low (98)
            {"open": 96, "high": 100, "low": 94, "close": 99},    # entry day
        ]
        triggers = smash_day_triggers(make_daily(rows), lookback=2)

        assert triggers["long_trigger"].iloc[3] == pytest.approx(107.0)
        assert triggers["long_trigger"].iloc[:3].isna().all()

    def test_short_arms_after_a_close_above_the_prior_range(self):
        """Yesterday closing over the prior N-day high arms a sell stop at its low."""
        rows = [
            {"open": 100, "high": 105, "low": 98, "close": 102},
            {"open": 102, "high": 106, "low": 99, "close": 104},
            {"open": 104, "high": 112, "low": 103, "close": 111},  # above the 2-day high (106)
            {"open": 111, "high": 113, "low": 108, "close": 110},
        ]
        triggers = smash_day_triggers(make_daily(rows), lookback=2)
        assert triggers["short_trigger"].iloc[3] == pytest.approx(103.0)


class TestTdom:
    """Trading-day-of-month seasonality."""

    def test_index_restarts_each_month(self):
        """The first trading day present in a month is index 1."""
        index = pd.DatetimeIndex(
            ["2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"], tz="US/Eastern"
        )
        tdom = trading_day_of_month(index)
        assert list(tdom) == [1, 1, 2, 1]

    def test_table_is_built_only_from_the_rows_it_is_given(self):
        """Fitting on a slice gives the same table as fitting on that slice of the whole."""
        df = make_random_daily(300)
        half = len(df) // 2

        from_slice = tdom_bias_table(df.iloc[:half], min_observations=3)
        from_copy = tdom_bias_table(df.iloc[:half].copy(), min_observations=3)
        assert from_slice == from_copy

        full = tdom_bias_table(df, min_observations=3)
        assert from_slice != full  # more data changes the means, as it must

    def test_flags_are_signs_of_the_fitted_table(self):
        """Bias flags are the sign of the table entry, 0 when absent."""
        index = pd.DatetimeIndex(["2024-02-01", "2024-02-02", "2024-02-05"], tz="US/Eastern")
        flags = tdom_bias_flags(index, {1: 5.0, 2: -3.0})
        assert list(flags) == [1, -1, 0]


class TestEmbargoSplit:
    """The chronological split with embargo gaps."""

    def test_segments_are_ordered_contiguous_and_cover_everything(self):
        """The five segments tile the timeline with no overlap or gap."""
        df = make_random_daily(1000)
        segments = build_embargo_split(df, 0.50, 0.20, embargo_days=5)

        assert [s.name for s in segments] == [
            "train", "embargo_1", "validation", "embargo_2", "oos"
        ]
        assert segments[0].start_pos == 0
        assert segments[-1].end_pos == len(df)
        for earlier, later in zip(segments, segments[1:]):
            assert earlier.end_pos == later.start_pos

    def test_embargo_length_and_fractions(self):
        """Embargoes are exactly the requested length; shares match the request."""
        df = make_random_daily(1000)
        segments = build_embargo_split(df, 0.50, 0.20, embargo_days=5)

        assert segment_by_name(segments, "embargo_1").n_days == 5
        assert segment_by_name(segments, "embargo_2").n_days == 5
        assert segment_by_name(segments, "train").n_days == 500
        assert segment_by_name(segments, "validation").n_days == 200
        assert segment_by_name(segments, "oos").n_days == 1000 - 500 - 200 - 10

    def test_raises_when_there_is_no_room_for_oos(self):
        """A frame too short for the split is rejected rather than silently trimmed."""
        with pytest.raises(ValueError):
            build_embargo_split(make_random_daily(12), 0.50, 0.20, embargo_days=5)


class TestDailyEngineFills:
    """Entry, exit and gap handling in the daily engine."""

    def _engine(self, **params) -> tuple[DailyBacktestEngine, WilliamsStrategy]:
        strategy = WilliamsStrategy(params=params)
        return DailyBacktestEngine(strategy, ZERO_COSTS, intraday=None), strategy

    def _signals(self, df: pd.DataFrame, longs=None, shorts=None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal": 0,
                "long_trigger": pd.Series(longs, index=df.index, dtype=float)
                if longs is not None
                else pd.Series(np.nan, index=df.index),
                "short_trigger": pd.Series(shorts, index=df.index, dtype=float)
                if shorts is not None
                else pd.Series(np.nan, index=df.index),
                "component": "test",
            },
            index=df.index,
        )

    def test_entry_fills_at_the_trigger_when_the_high_reaches_it(self):
        """A buy stop above the open fills at the trigger price."""
        rows = [
            {"open": 100, "high": 120, "low": 99, "close": 118},
            {"open": 118, "high": 119, "low": 117, "close": 118},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=1)
        result = engine.run(df, self._signals(df, longs=[110.0, np.nan]))

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(110.0)
        assert trade.direction == 1
        assert trade.exit_reason == "time_exit"
        assert trade.exit_price == pytest.approx(118.0)

    def test_entry_fills_at_the_open_when_the_session_gaps_past_the_trigger(self):
        """A stop order cannot fill better than the open."""
        rows = [
            {"open": 130, "high": 135, "low": 128, "close": 132},
            {"open": 132, "high": 133, "low": 131, "close": 132},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=1)
        result = engine.run(df, self._signals(df, longs=[110.0, np.nan]))

        assert result.trades[0].entry_price == pytest.approx(130.0)

    def test_no_entry_when_the_trigger_is_never_reached(self):
        """A resting stop that the session never touches does not fill."""
        rows = [
            {"open": 100, "high": 105, "low": 99, "close": 104},
            {"open": 104, "high": 106, "low": 103, "close": 105},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=1)
        result = engine.run(df, self._signals(df, longs=[110.0, np.nan]))
        assert result.trades == []

    def test_overnight_gap_through_the_stop_fills_at_the_open(self):
        """The headline gap case: the fill is the open, not the stop."""
        rows = [
            # Enter long at 1000, stop at 950, target 1100. Day closes at 1010.
            {"open": 990, "high": 1020, "low": 985, "close": 1010},
            # Next session opens at 900, far below the 950 stop.
            {"open": 900, "high": 905, "low": 880, "close": 890},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=5)
        result = engine.run(df, self._signals(df, longs=[1000.0, np.nan]))

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "gap_stop"
        assert trade.gap_fill is True
        assert trade.exit_price == pytest.approx(900.0)
        assert trade.intended_stop == pytest.approx(950.0)
        # 50 points worse than the intended stop, and the loss is 100 not 50.
        assert trade.gap_slippage_points == pytest.approx(50.0)
        assert trade.pnl_net == pytest.approx(-100.0)

    def test_short_gap_through_the_stop_fills_at_the_open(self):
        """Mirror case for a short position."""
        rows = [
            {"open": 1010, "high": 1015, "low": 980, "close": 990},
            {"open": 1100, "high": 1110, "low": 1095, "close": 1105},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=5)
        result = engine.run(df, self._signals(df, shorts=[1000.0, np.nan]))

        trade = result.trades[0]
        assert trade.exit_reason == "gap_stop"
        assert trade.exit_price == pytest.approx(1100.0)
        assert trade.pnl_net == pytest.approx(-100.0)

    def test_gap_beyond_the_stop_is_not_recorded_as_a_normal_stop(self):
        """An open exactly at the stop is a normal stop, not a gap fill."""
        rows = [
            {"open": 990, "high": 1020, "low": 985, "close": 1010},
            {"open": 950, "high": 960, "low": 940, "close": 955},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=5)
        result = engine.run(df, self._signals(df, longs=[1000.0, np.nan]))

        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        assert trade.gap_fill is False
        assert trade.pnl_net == pytest.approx(-50.0)

    def test_max_hold_days_one_closes_on_the_entry_day(self):
        """max_hold_days=1 carries no overnight risk."""
        rows = [
            {"open": 990, "high": 1020, "low": 985, "close": 1010},
            {"open": 900, "high": 905, "low": 880, "close": 890},
        ]
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=50, target_points=100, max_hold_days=1)
        result = engine.run(df, self._signals(df, longs=[1000.0, np.nan]))

        trade = result.trades[0]
        assert trade.hold_days == 1
        assert trade.exit_reason == "time_exit"
        assert trade.exit_price == pytest.approx(1010.0)

    def test_time_exit_after_the_configured_number_of_days(self):
        """A position with no level touched closes at the Nth day's close."""
        rows = [{"open": 1000, "high": 1005, "low": 995, "close": 1000}] * 6
        rows[0] = {"open": 995, "high": 1005, "low": 990, "close": 1000}
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=500, target_points=1000, max_hold_days=3)
        longs = [1000.0] + [np.nan] * 5
        result = engine.run(df, self._signals(df, longs=longs))

        trade = result.trades[0]
        assert trade.exit_reason == "time_exit"
        assert trade.hold_days == 3
        assert trade.exit_idx == 2

    def test_trailing_stop_ratchets_on_daily_closes(self):
        """The optional trailing exit follows the closes and then stops the trade out."""
        rows = [
            {"open": 995, "high": 1005, "low": 990, "close": 1100},   # enter 1000, close 1100
            {"open": 1100, "high": 1105, "low": 1000, "close": 1010},  # trail at 1080 is hit
        ]
        df = make_daily(rows)
        engine, _ = self._engine(
            stop_points=500, target_points=1000, max_hold_days=5, trailing_stop_points=20
        )
        result = engine.run(df, self._signals(df, longs=[1000.0, np.nan]))

        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        assert trade.exit_price == pytest.approx(1080.0)

    def test_position_is_closed_at_the_end_of_the_span(self):
        """close_at_end stops a trade straddling a split boundary."""
        rows = [{"open": 995, "high": 1005, "low": 990, "close": 1000}] * 4
        df = make_daily(rows)
        engine, _ = self._engine(stop_points=500, target_points=1000, max_hold_days=10)
        longs = [1000.0] + [np.nan] * 3
        result = engine.run(df, self._signals(df, longs=longs), start=0, end=2)

        assert result.trades[0].exit_reason == "span_end"
        assert result.trades[0].exit_idx == 1

    def test_only_the_span_is_simulated(self):
        """Days outside [start, end) produce neither trades nor daily P&L."""
        df = make_random_daily(60)
        strategy = WilliamsStrategy(params={"stop_points": 50, "target_points": 100,
                                            "max_hold_days": 3})
        engine = DailyBacktestEngine(strategy, ZERO_COSTS, intraday=None)
        signals = strategy.generate_signals(df)

        result = engine.run(df, signals, start=30, end=45)
        assert len(result.daily_pnl_points) == 15
        assert result.daily_pnl_points.index[0] == df.index[30]
        for trade in result.trades:
            assert 30 <= trade.entry_idx < 45

    def test_daily_pnl_sums_to_realized_pnl(self):
        """Mark-to-market days add up to the realized trade P&L over the span."""
        df = make_random_daily(200)
        strategy = WilliamsStrategy(params={"stop_points": 60, "target_points": 120,
                                            "max_hold_days": 3})
        engine = DailyBacktestEngine(strategy, ZERO_COSTS, intraday=None)
        signals = strategy.generate_signals(df)
        result = engine.run(df, signals, start=20, end=180)

        assert result.daily_pnl_points.sum() == pytest.approx(
            sum(t.pnl_net for t in result.trades), abs=1e-6
        )

    def test_no_overlapping_positions(self):
        """Only one position is open at a time."""
        df = make_random_daily(250)
        strategy = WilliamsStrategy(params={"stop_points": 60, "target_points": 120,
                                            "max_hold_days": 5})
        engine = DailyBacktestEngine(strategy, ZERO_COSTS, intraday=None)
        result = engine.run(df, strategy.generate_signals(df))

        for earlier, later in zip(result.trades, result.trades[1:]):
            assert later.entry_idx >= earlier.exit_idx


class TestGapStatistics:
    """Gap-fill reporting."""

    def test_counts_and_averages_only_losing_gap_fills(self):
        """The share is measured against losing trades, and slippage averages gap fills."""
        rows = [
            {"open": 990, "high": 1020, "low": 985, "close": 1010},
            {"open": 900, "high": 905, "low": 880, "close": 890},
        ]
        df = make_daily(rows)
        strategy = WilliamsStrategy(params={"stop_points": 50, "target_points": 100,
                                            "max_hold_days": 5})
        engine = DailyBacktestEngine(strategy, ZERO_COSTS, intraday=None)
        signals = pd.DataFrame(
            {
                "signal": 0,
                "long_trigger": pd.Series([1000.0, np.nan], index=df.index),
                "short_trigger": pd.Series([np.nan, np.nan], index=df.index),
                "component": "test",
            },
            index=df.index,
        )
        stats = gap_fill_statistics(engine.run(df, signals).trades)

        assert stats["losing_trades"] == 1
        assert stats["gap_fill_losers"] == 1
        assert stats["gap_fill_share_of_losers"] == pytest.approx(1.0)
        assert stats["avg_gap_slippage_points"] == pytest.approx(50.0)


class TestPropAccountReplay:
    """The in-loop prop firm replay."""

    def test_daily_units_are_one_per_day(self):
        """Every trading day becomes exactly one resamplable unit."""
        series = pd.Series(
            [1.0, -2.0, 0.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        units = daily_pnl_units(series)
        assert len(units) == 3
        assert [u.pnl_net for u in units] == [1.0, -2.0, 0.0]

    def test_daily_limit_breach_kills_the_account_inside_the_loop(self):
        """A single day past the limit ends the replay at that day's equity."""
        series = pd.Series(
            [10.0, -60.0, 100.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        result = replay_prop_account(
            daily_pnl_points=series,
            dollars_per_point=20.0,
            account_size=50_000.0,
            daily_loss_limit=1_000.0,
            equity_floor=48_000.0,
            profit_target=3_000.0,
        )

        assert result["survived"] is False
        assert result["breach_reason"] == "daily loss limit"
        assert result["breach_day"] == series.index[1]
        # The third day's +100 points is never earned: the account was dead.
        assert result["final_equity"] == pytest.approx(50_000 + 200 - 1200)

    def test_equity_floor_breach_is_also_a_hard_fail(self):
        """Dropping under the floor ends the replay even inside the daily limit."""
        series = pd.Series(
            [-40.0, -40.0, -40.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        result = replay_prop_account(
            daily_pnl_points=series,
            dollars_per_point=20.0,
            account_size=50_000.0,
            daily_loss_limit=1_000.0,
            equity_floor=48_000.0,
            profit_target=3_000.0,
        )

        assert result["survived"] is False
        assert result["breach_reason"] == "equity floor"

    def test_a_clean_run_survives_and_records_the_target(self):
        """No breach means survival, and the target flag reflects peak equity."""
        series = pd.Series(
            [40.0] * 5,
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )
        result = replay_prop_account(
            daily_pnl_points=series,
            dollars_per_point=20.0,
            account_size=50_000.0,
            daily_loss_limit=1_000.0,
            equity_floor=48_000.0,
            profit_target=3_000.0,
        )

        assert result["survived"] is True
        assert result["reached_target"] is True
        assert result["final_equity"] == pytest.approx(54_000.0)


class TestStrategyComposition:
    """Component selection and combination inside the strategy."""

    def test_single_component_matches_its_indicator(self):
        """With one component the strategy just forwards that indicator's triggers."""
        df = make_random_daily(150)
        strategy = WilliamsStrategy(params={"components": ("oops",)})
        signals = strategy.generate_signals(df)
        expected = oops_triggers(df)

        pd.testing.assert_series_equal(
            signals["long_trigger"], expected["long_trigger"], check_names=False
        )

    def test_combination_takes_the_nearest_resting_stop(self):
        """With two components the lower buy stop wins, as it would fill first."""
        df = make_random_daily(200)
        combined = WilliamsStrategy(
            params={"components": ("gsv", "oops"), "gsv_lookback": 5, "gsv_multiplier": 0.8}
        ).generate_signals(df)

        gsv = gsv_triggers(df, 5, 0.8)["long_trigger"]
        oops = oops_triggers(df)["long_trigger"]
        expected = pd.concat([gsv, oops], axis=1).min(axis=1)

        pd.testing.assert_series_equal(
            combined["long_trigger"], expected, check_names=False
        )

    def test_tdom_filter_removes_triggers_on_days_without_the_bias(self):
        """The filter can only remove triggers, never add or move one."""
        df = make_random_daily(200)
        table = tdom_bias_table(df.iloc[:120], min_observations=3)

        unfiltered = WilliamsStrategy(
            params={"components": ("gsv",), "tdom_bias": table}
        ).generate_signals(df)
        filtered = WilliamsStrategy(
            params={"components": ("gsv",), "tdom_bias": table, "tdom_filter": True}
        ).generate_signals(df)

        kept = filtered["long_trigger"].notna()
        assert kept.sum() < unfiltered["long_trigger"].notna().sum()
        pd.testing.assert_series_equal(
            filtered["long_trigger"][kept], unfiltered["long_trigger"][kept]
        )

    def test_unknown_component_is_rejected(self):
        """A typo in the component list fails loudly."""
        df = make_random_daily(60)
        with pytest.raises(ValueError, match="Unknown Williams component"):
            WilliamsStrategy(params={"components": ("momentum",)}).generate_signals(df)
