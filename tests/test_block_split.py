"""Tests for the interleaved-block train/validation/OOS split.

Covers block construction, round-robin assignment, the guarantee that no
trade straddles a block boundary, and the coverage/share of each split.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester.block_split import (
    DEFAULT_ASSIGNMENT,
    blocks_for,
    build_blocks,
    month_confounding_warning,
    simulate_blocks,
)
from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine
from src.strategies.simple_strategy import SimpleStrategy


def make_session_data(months: int = 8, bars_per_day: int = 6) -> pd.DataFrame:
    """Create 5-min session bars spanning several calendar months.

    Args:
        months: Number of consecutive months to cover.
        bars_per_day: Bars generated per weekday.

    Returns:
        DataFrame with OHLCV and delta columns on a DatetimeIndex.
    """
    days = pd.date_range("2021-01-01", periods=months * 31, freq="B")
    stamps = []
    for day in days:
        if day.month > months:
            break
        for k in range(bars_per_day):
            stamps.append(day + pd.Timedelta(minutes=30 + 5 * k, hours=9))

    index = pd.DatetimeIndex(stamps)
    n = len(index)
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(0, 3.0, n)) + 15000

    return pd.DataFrame(
        {
            "open": base,
            "high": base + 5.0,
            "low": base - 5.0,
            "close": base,
            "volume": rng.uniform(1000, 4000, n),
            "delta": rng.normal(0, 500, n),
        },
        index=index,
    )


class TestBuildBlocks:
    """Tests for block construction and assignment."""

    def test_one_block_per_calendar_month(self):
        """Each calendar month becomes exactly one block."""
        df = make_session_data(months=6)
        blocks = build_blocks(df, block_months=1)

        assert len(blocks) == df.index.to_period("M").nunique()

    def test_round_robin_assignment(self):
        """Blocks cycle train, train, validation, oos."""
        df = make_session_data(months=8)
        blocks = build_blocks(df, block_months=1)

        assert [b.split for b in blocks[:4]] == list(DEFAULT_ASSIGNMENT)
        for b in blocks:
            assert b.split == DEFAULT_ASSIGNMENT[b.idx % 4]

    def test_blocks_are_contiguous_and_cover_all_bars(self):
        """Blocks partition the data with no gaps and no overlap."""
        df = make_session_data(months=8)
        blocks = build_blocks(df, block_months=1)

        assert blocks[0].start_pos == 0
        assert blocks[-1].end_pos == len(df)
        for prev, nxt in zip(blocks, blocks[1:]):
            assert prev.end_pos == nxt.start_pos
        assert sum(b.n_bars for b in blocks) == len(df)

    def test_splits_are_disjoint(self):
        """No bar belongs to more than one split."""
        df = make_session_data(months=8)
        blocks = build_blocks(df, block_months=1)

        seen: set[int] = set()
        for b in blocks:
            positions = set(range(b.start_pos, b.end_pos))
            assert not (positions & seen)
            seen |= positions

    def test_split_shares_are_roughly_50_25_25(self):
        """Round-robin gives about 50% train, 25% validation, 25% OOS."""
        df = make_session_data(months=12)
        blocks = build_blocks(df, block_months=1)
        total = sum(b.n_bars for b in blocks)

        train = sum(b.n_bars for b in blocks_for(blocks, "train")) / total
        val = sum(b.n_bars for b in blocks_for(blocks, "validation")) / total
        oos = sum(b.n_bars for b in blocks_for(blocks, "oos")) / total

        assert train == pytest.approx(0.5, abs=0.08)
        assert val == pytest.approx(0.25, abs=0.08)
        assert oos == pytest.approx(0.25, abs=0.08)

    def test_every_split_spans_multiple_years(self):
        """Interleaving puts every split in more than one calendar year."""
        days = pd.date_range("2021-01-04", "2024-09-30", freq="B")
        index = pd.DatetimeIndex([d + pd.Timedelta(hours=10) for d in days])
        df = pd.DataFrame(
            {
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "delta": 0.0,
            },
            index=index,
        )
        blocks = build_blocks(df, block_months=1)

        for split in ("train", "validation", "oos"):
            years = {b.start_time.year for b in blocks_for(blocks, split)}
            assert years == {2021, 2022, 2023, 2024}

    def test_empty_dataframe(self):
        """An empty frame yields no blocks."""
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        assert build_blocks(empty) == []

    def test_timezone_aware_index_supported(self):
        """A tz-aware index is handled without dropping bars."""
        df = make_session_data(months=4)
        df.index = df.index.tz_localize("US/Eastern")

        blocks = build_blocks(df, block_months=1)
        assert sum(b.n_bars for b in blocks) == len(df)


class TestMonthConfounding:
    """Tests for the calendar-month confounding warning."""

    def test_warns_when_splits_never_share_a_month(self):
        """A 4-block cycle over 12 months gives each split fixed months."""
        df = make_session_data(months=12)
        blocks = build_blocks(df, block_months=1)

        warning = month_confounding_warning(blocks, ("train", "validation", "oos"))

        assert "WARNING" in warning
        assert "confounded" in warning

    def test_no_warning_when_months_are_shared(self):
        """A cycle that does not divide the year shares months across splits."""
        df = make_session_data(months=12)
        blocks = build_blocks(
            df,
            block_months=1,
            assignment=("train", "train", "train", "validation", "oos"),
        )

        warning = month_confounding_warning(blocks, ("train", "validation", "oos"))

        assert warning == ""


class TestSimulateBlocks:
    """Tests that block simulation never lets a trade cross a boundary."""

    @staticmethod
    def _engine(strategy) -> BacktestEngine:
        return BacktestEngine(
            strategy,
            CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0),
            exit_management={
                "stop_loss_mode": "fixed",
                "partial_close_enabled": False,
            },
            close_at_end=True,
        )

    def test_no_trade_straddles_a_block_boundary(self):
        """Entry and exit of every trade sit inside the same block."""
        df = make_session_data(months=10)
        strategy = SimpleStrategy(
            params={
                "profile_lookback": 20,
                "level_proximity_points": 10,
                "delta_threshold": 0.5,
                "stop_points": 20,
                "target_points": 30,
            }
        )
        engine = self._engine(strategy)
        signals = strategy.generate_signals(df)["signal"]
        blocks = build_blocks(df, block_months=1)
        train_blocks = blocks_for(blocks, "train")

        result = simulate_blocks(engine, df, signals, train_blocks)

        assert len(result.trades) > 0
        ranges = [(b.start_time, b.end_time) for b in train_blocks]
        for trade in result.trades:
            inside = [
                start <= trade.entry_time and trade.exit_time <= end
                for start, end in ranges
            ]
            assert sum(inside) == 1, "Trade crossed a block boundary"

    def test_trades_only_come_from_the_requested_split(self):
        """A split's trades all start inside that split's blocks."""
        df = make_session_data(months=10)
        strategy = SimpleStrategy(
            params={
                "profile_lookback": 20,
                "level_proximity_points": 10,
                "delta_threshold": 0.5,
                "stop_points": 20,
                "target_points": 30,
            }
        )
        engine = self._engine(strategy)
        signals = strategy.generate_signals(df)["signal"]
        blocks = build_blocks(df, block_months=1)
        oos_blocks = blocks_for(blocks, "oos")

        result = simulate_blocks(engine, df, signals, oos_blocks)

        allowed = set()
        for b in oos_blocks:
            allowed |= set(df.index[b.start_pos : b.end_pos])
        for trade in result.trades:
            assert trade.entry_time in allowed

    def test_equity_curve_matches_trade_pnl(self):
        """The stitched equity curve ends at the sum of net trade P&L."""
        df = make_session_data(months=10)
        strategy = SimpleStrategy(
            params={
                "profile_lookback": 20,
                "level_proximity_points": 10,
                "delta_threshold": 0.5,
                "stop_points": 20,
                "target_points": 30,
            }
        )
        engine = self._engine(strategy)
        signals = strategy.generate_signals(df)["signal"]
        blocks = build_blocks(df, block_months=1)
        val_blocks = blocks_for(blocks, "validation")

        result = simulate_blocks(engine, df, signals, val_blocks)
        expected = sum(t.pnl_net for t in result.trades)

        assert float(result.equity_net.iloc[-1]) == pytest.approx(expected)
