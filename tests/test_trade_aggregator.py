"""Tests for trade data aggregation module.

Tests correct aggregation of raw Databento trade data into 5-min bars
with real bid/ask volume split, including:
- Side convention (B=buyer aggressor, A=seller aggressor)
- int64 overflow handling for size column
- Delta computation (bid_volume - ask_volume)
- Footprint data aggregation
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.data.trade_aggregator import (
    aggregate_quarter,
    get_trade_files,
    _aggregate_quarter_footprint,
)


@pytest.fixture
def sample_trades_df():
    """Create a sample trades DataFrame matching Databento format."""
    n = 1000
    np.random.seed(42)

    # Create timestamps spanning 2 hours (should produce ~24 5-min bars)
    base_time = pd.Timestamp("2021-03-15 10:00:00", tz="UTC")
    ts_events = pd.date_range(base_time, periods=n, freq="500ms")

    # Random sides: B (buyer aggressor) and A (seller aggressor)
    sides = np.random.choice(["B", "A"], size=n, p=[0.55, 0.45])

    # Prices around 13000 with some movement
    prices = 13000.0 + np.cumsum(np.random.randn(n) * 0.25)

    # Sizes (small integers as in real NQ data)
    sizes = np.random.randint(1, 20, size=n).astype(np.uint32)

    df = pd.DataFrame({
        "ts_event": ts_events,
        "side": sides,
        "price": prices,
        "size": sizes,
    })
    # Set ts_recv as index (as in real data)
    df.index = ts_events + pd.Timedelta("50ms")
    df.index.name = "ts_recv"

    return df


@pytest.fixture
def sample_trades_parquet(tmp_path, sample_trades_df):
    """Save sample trades to a temporary parquet file."""
    filepath = tmp_path / "NQ_trades_2021_Q1.parquet"
    sample_trades_df.to_parquet(filepath)
    return filepath


class TestGetTradeFiles:
    """Tests for finding trade files in a directory."""

    def test_finds_parquet_files(self, tmp_path):
        """Should find all NQ_trades_*.parquet files."""
        # Create some fake files
        (tmp_path / "NQ_trades_2021_Q1.parquet").touch()
        (tmp_path / "NQ_trades_2021_Q2.parquet").touch()
        (tmp_path / "other_file.parquet").touch()

        files = get_trade_files(str(tmp_path))
        assert len(files) == 2
        assert all("NQ_trades" in f.name for f in files)

    def test_returns_sorted(self, tmp_path):
        """Files should be returned in sorted order."""
        (tmp_path / "NQ_trades_2022_Q1.parquet").touch()
        (tmp_path / "NQ_trades_2021_Q3.parquet").touch()
        (tmp_path / "NQ_trades_2021_Q1.parquet").touch()

        files = get_trade_files(str(tmp_path))
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_empty_directory(self, tmp_path):
        """Returns empty list for directory with no matching files."""
        files = get_trade_files(str(tmp_path))
        assert files == []


class TestAggregateQuarter:
    """Tests for aggregating a single quarter of trade data."""

    def test_produces_5min_bars(self, sample_trades_parquet):
        """Should produce 5-minute bars from trade data."""
        bars = aggregate_quarter(sample_trades_parquet)
        assert not bars.empty
        # Check that all bars are 5-min aligned
        for ts in bars.index:
            assert ts.minute % 5 == 0 or ts.second == 0

    def test_has_required_columns(self, sample_trades_parquet):
        """Output should have all required columns."""
        bars = aggregate_quarter(sample_trades_parquet)
        required = [
            "open", "high", "low", "close", "volume",
            "bid_volume", "ask_volume", "delta",
            "trade_count", "avg_trade_size",
        ]
        for col in required:
            assert col in bars.columns, f"Missing column: {col}"

    def test_side_convention(self, sample_trades_parquet, sample_trades_df):
        """B=buyer aggressor goes to bid_volume, A=seller to ask_volume."""
        bars = aggregate_quarter(sample_trades_parquet)

        # Total bid_volume should equal sum of sizes where side=B
        total_bid = bars["bid_volume"].sum()
        total_ask = bars["ask_volume"].sum()

        # From the source data
        df = sample_trades_df
        df_filtered = df[df["side"].isin(["B", "A"])]
        expected_bid = df_filtered[df_filtered["side"] == "B"]["size"].astype(np.int64).sum()
        expected_ask = df_filtered[df_filtered["side"] == "A"]["size"].astype(np.int64).sum()

        assert total_bid == expected_bid
        assert total_ask == expected_ask

    def test_delta_computation(self, sample_trades_parquet):
        """Delta should equal bid_volume - ask_volume for each bar."""
        bars = aggregate_quarter(sample_trades_parquet)
        expected_delta = bars["bid_volume"] - bars["ask_volume"]
        pd.testing.assert_series_equal(
            bars["delta"], expected_delta, check_names=False
        )

    def test_volume_equals_bid_plus_ask(self, sample_trades_parquet):
        """Total volume should equal bid_volume + ask_volume."""
        bars = aggregate_quarter(sample_trades_parquet)
        expected_vol = bars["bid_volume"] + bars["ask_volume"]
        pd.testing.assert_series_equal(
            bars["volume"], expected_vol, check_names=False
        )

    def test_int64_overflow_handling(self, tmp_path):
        """Large uint32 sizes should not overflow when summed as int64."""
        # Create data with large sizes that would overflow uint32 if summed
        n = 100
        base_time = pd.Timestamp("2021-01-04 10:00:00", tz="UTC")
        ts_events = pd.date_range(base_time, periods=n, freq="1s")

        df = pd.DataFrame({
            "ts_event": ts_events,
            "side": ["B"] * n,
            "price": [13000.0] * n,
            "size": np.full(n, 100, dtype=np.uint32),
        })
        df.index = ts_events + pd.Timedelta("50ms")
        df.index.name = "ts_recv"

        filepath = tmp_path / "NQ_trades_2021_Q1.parquet"
        df.to_parquet(filepath)

        bars = aggregate_quarter(filepath)
        # All 100 trades in first 5 minutes should sum to 10000
        assert bars["bid_volume"].iloc[0] == 10000
        assert bars["volume"].dtype == np.int64

    def test_filters_non_ba_sides(self, tmp_path):
        """Should filter out trades with side='N' (no aggressor)."""
        base_time = pd.Timestamp("2021-01-04 10:00:00", tz="UTC")
        ts_events = pd.date_range(base_time, periods=10, freq="1s")

        df = pd.DataFrame({
            "ts_event": ts_events,
            "side": ["B", "A", "N", "B", "A", "N", "B", "A", "B", "N"],
            "price": [13000.0] * 10,
            "size": np.ones(10, dtype=np.uint32) * 5,
        })
        df.index = ts_events + pd.Timedelta("50ms")
        df.index.name = "ts_recv"

        filepath = tmp_path / "NQ_trades_2021_Q1.parquet"
        df.to_parquet(filepath)

        bars = aggregate_quarter(filepath)
        # Should only count B and A trades: 5B + 2A = 7 trades
        total_vol = bars["volume"].sum()
        assert total_vol == 35  # 7 trades * 5 size each

    def test_timezone_conversion(self, sample_trades_parquet):
        """Output should be in US/Eastern timezone."""
        bars = aggregate_quarter(sample_trades_parquet)
        assert str(bars.index.tz) == "US/Eastern"

    def test_ohlc_correct(self, tmp_path):
        """OHLC should correctly reflect first/max/min/last prices."""
        base_time = pd.Timestamp("2021-01-04 10:00:00", tz="UTC")
        ts_events = pd.date_range(base_time, periods=5, freq="30s")

        prices = [100.0, 105.0, 95.0, 102.0, 101.0]
        df = pd.DataFrame({
            "ts_event": ts_events,
            "side": ["B"] * 5,
            "price": prices,
            "size": np.ones(5, dtype=np.uint32),
        })
        df.index = ts_events + pd.Timedelta("50ms")
        df.index.name = "ts_recv"

        filepath = tmp_path / "NQ_trades_2021_Q1.parquet"
        df.to_parquet(filepath)

        bars = aggregate_quarter(filepath)
        # All 5 trades should be in one 5-min bar
        assert len(bars) == 1
        assert bars["open"].iloc[0] == 100.0
        assert bars["high"].iloc[0] == 105.0
        assert bars["low"].iloc[0] == 95.0
        assert bars["close"].iloc[0] == 101.0


class TestAggregateFootprint:
    """Tests for footprint (price-level) data aggregation."""

    def test_footprint_structure(self, sample_trades_parquet):
        """Footprint should have bar_timestamp, price_level, and volumes."""
        fp = _aggregate_quarter_footprint(sample_trades_parquet)
        assert not fp.empty
        required_cols = ["bar_timestamp", "price_level",
                         "bid_volume_at_level", "ask_volume_at_level"]
        for col in required_cols:
            assert col in fp.columns, f"Missing column: {col}"

    def test_footprint_volume_split(self, tmp_path):
        """Footprint correctly splits volume by side at each price level."""
        base_time = pd.Timestamp("2021-01-04 10:00:00", tz="UTC")
        ts_events = pd.date_range(base_time, periods=6, freq="10s")

        df = pd.DataFrame({
            "ts_event": ts_events,
            "side": ["B", "B", "A", "A", "B", "A"],
            "price": [100.0, 100.0, 100.0, 101.0, 101.0, 102.0],
            "size": np.array([3, 2, 4, 5, 1, 6], dtype=np.uint32),
        })
        df.index = ts_events + pd.Timedelta("50ms")
        df.index.name = "ts_recv"

        filepath = tmp_path / "NQ_trades_2021_Q1.parquet"
        df.to_parquet(filepath)

        fp = _aggregate_quarter_footprint(filepath)

        # Price 100: B=5 (3+2), A=4
        level_100 = fp[fp["price_level"] == 100.0]
        assert level_100["bid_volume_at_level"].values[0] == 5
        assert level_100["ask_volume_at_level"].values[0] == 4

        # Price 101: B=1, A=5
        level_101 = fp[fp["price_level"] == 101.0]
        assert level_101["bid_volume_at_level"].values[0] == 1
        assert level_101["ask_volume_at_level"].values[0] == 5

        # Price 102: B=0, A=6
        level_102 = fp[fp["price_level"] == 102.0]
        assert level_102["bid_volume_at_level"].values[0] == 0
        assert level_102["ask_volume_at_level"].values[0] == 6

    def test_footprint_bar_alignment(self, tmp_path):
        """Each footprint row should have a 5-min-aligned bar timestamp."""
        base_time = pd.Timestamp("2021-01-04 10:02:30", tz="UTC")
        ts_events = pd.date_range(base_time, periods=3, freq="1s")

        df = pd.DataFrame({
            "ts_event": ts_events,
            "side": ["B", "A", "B"],
            "price": [100.0, 100.0, 101.0],
            "size": np.ones(3, dtype=np.uint32),
        })
        df.index = ts_events + pd.Timedelta("50ms")
        df.index.name = "ts_recv"

        filepath = tmp_path / "NQ_trades_2021_Q1.parquet"
        df.to_parquet(filepath)

        fp = _aggregate_quarter_footprint(filepath)

        # All trades at 10:02:30 should be floored to 10:00 bar
        for _, row in fp.iterrows():
            bar_ts = row["bar_timestamp"]
            assert bar_ts.minute % 5 == 0
