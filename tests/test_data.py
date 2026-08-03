"""Tests for configuration loading and data preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, get_data_config, get_costs_config, get_monte_carlo_config
from src.data.preprocessor import (
    compute_cumulative_delta,
    compute_relative_volume,
    compute_volume_delta,
    compute_vwap,
    preprocess,
)


# --- Fixtures ---


@pytest.fixture
def sample_ohlcv():
    """Create a sample OHLCV DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(42)

    # Generate realistic-looking price data
    close = 15000 + np.cumsum(np.random.randn(50) * 50)
    open_ = close + np.random.randn(50) * 20
    high = np.maximum(open_, close) + np.abs(np.random.randn(50) * 30)
    low = np.minimum(open_, close) - np.abs(np.random.randn(50) * 30)
    volume = np.random.randint(50000, 200000, size=50).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


@pytest.fixture
def known_ohlcv():
    """Create a DataFrame with known values for exact verification."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0, 103.0, 102.0],
            "high": [105.0, 104.0, 103.0, 106.0, 105.0],
            "low": [99.0, 100.0, 99.0, 101.0, 100.0],
            "close": [103.0, 101.0, 102.0, 105.0, 103.0],
            "volume": [1000.0, 1500.0, 800.0, 2000.0, 1200.0],
        },
        index=dates,
    )
    return df


# --- Config Tests ---


class TestConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test that default config loads successfully."""
        config = load_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_config_has_data_section(self):
        """Test that config contains data section with required keys."""
        config = load_config()
        data = get_data_config(config)
        assert "source" in data
        assert "data_file" in data
        assert "train_ratio" in data
        assert "train_split" in data
        assert "databento" in data
        assert "yfinance" in data

    def test_config_has_costs_section(self):
        """Test that config contains costs section with required keys."""
        config = load_config()
        costs = get_costs_config(config)
        assert "slippage_points" in costs
        assert "commission_per_round_trip" in costs
        assert "point_value" in costs

    def test_config_has_monte_carlo_section(self):
        """Test that config contains Monte Carlo section."""
        config = load_config()
        mc = get_monte_carlo_config(config)
        assert "simulations" in mc
        assert "confidence_level" in mc
        assert "ruin_threshold" in mc

    def test_config_has_strategy_sections(self):
        """Test that config contains strategy-specific sections."""
        config = load_config()
        assert "order_flow_strategy" in config
        assert "volume_profile_strategy" in config

    def test_train_split_valid_range(self):
        """Test that train_ratio is between 0 and 1."""
        config = load_config()
        data = get_data_config(config)
        assert 0 < data["train_ratio"] < 1


# --- Preprocessor Tests ---


class TestVolumeDelta:
    """Tests for volume delta computation."""

    def test_bullish_bar_positive_delta(self, known_ohlcv):
        """Bullish bars (close > open) should have positive volume delta."""
        delta = compute_volume_delta(known_ohlcv)
        # Bar 0: close=103, open=100, so bullish -> positive delta
        assert delta.iloc[0] > 0

    def test_bearish_bar_negative_delta(self, known_ohlcv):
        """Bearish bars (close < open) should have negative volume delta."""
        delta = compute_volume_delta(known_ohlcv)
        # Bar 1: close=101, open=102, so bearish -> negative delta
        assert delta.iloc[1] < 0

    def test_volume_delta_formula(self, known_ohlcv):
        """Verify exact volume delta calculation."""
        delta = compute_volume_delta(known_ohlcv)
        # Bar 0: volume=1000, (close-open)/(high-low) = (103-100)/(105-99) = 3/6 = 0.5
        # delta = 1000 * 0.5 = 500
        expected = 1000.0 * (103.0 - 100.0) / (105.0 - 99.0)
        assert abs(delta.iloc[0] - expected) < 1e-10

    def test_doji_bar_zero_delta(self):
        """A doji bar (high == low) should have zero delta."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        delta = compute_volume_delta(df)
        assert delta.iloc[0] == 0


class TestCumulativeDelta:
    """Tests for cumulative delta computation."""

    def test_cumulative_delta_is_running_sum(self, known_ohlcv):
        """For daily data, cumulative delta should be running sum of volume delta."""
        delta = compute_volume_delta(known_ohlcv)
        cum_delta = compute_cumulative_delta(known_ohlcv)
        # Cumulative at bar N should equal sum of deltas 0..N
        for i in range(len(known_ohlcv)):
            expected = delta.iloc[: i + 1].sum()
            assert abs(cum_delta.iloc[i] - expected) < 1e-6

    def test_cumulative_delta_shape(self, sample_ohlcv):
        """Cumulative delta should have same length as input."""
        cum_delta = compute_cumulative_delta(sample_ohlcv)
        assert len(cum_delta) == len(sample_ohlcv)


class TestRelativeVolume:
    """Tests for relative volume computation."""

    def test_relative_volume_average_near_one(self, sample_ohlcv):
        """Average relative volume should be approximately 1.0."""
        rel_vol = compute_relative_volume(sample_ohlcv, window=20)
        # After warmup period, mean should be close to 1
        assert 0.5 < rel_vol.iloc[20:].mean() < 1.5

    def test_relative_volume_spike_detection(self):
        """A volume spike should produce relative volume > 1."""
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        volumes = [100.0] * 24 + [500.0]  # Spike at end
        df = pd.DataFrame(
            {
                "open": [100.0] * 25,
                "high": [101.0] * 25,
                "low": [99.0] * 25,
                "close": [100.5] * 25,
                "volume": volumes,
            },
            index=dates,
        )
        rel_vol = compute_relative_volume(df, window=20)
        # Last bar should have high relative volume
        assert rel_vol.iloc[-1] > 2.0


class TestVWAP:
    """Tests for VWAP calculation."""

    def test_vwap_single_bar(self):
        """VWAP of a single bar should equal its typical price."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        vwap = compute_vwap(df)
        typical_price = (105.0 + 95.0 + 102.0) / 3
        assert abs(vwap.iloc[0] - typical_price) < 1e-10

    def test_vwap_weighted_by_volume(self):
        """VWAP should be weighted toward high-volume bars."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 110.0, 105.0],
                "high": [100.0, 110.0, 105.0],
                "low": [100.0, 110.0, 105.0],
                "close": [100.0, 110.0, 105.0],
                "volume": [100.0, 900.0, 100.0],
            },
            index=dates,
        )
        vwap = compute_vwap(df)
        # Bar 2 (index 1) has 10x more volume at price 110
        # So cumulative VWAP after bar 2 should be closer to 110 than 100
        # TP_1=100, TP_2=110: VWAP_2 = (100*100 + 110*900) / (100+900) = 109
        expected_vwap_2 = (100.0 * 100 + 110.0 * 900) / (100 + 900)
        assert abs(vwap.iloc[1] - expected_vwap_2) < 1e-10

    def test_vwap_shape(self, sample_ohlcv):
        """VWAP should have same length as input data."""
        vwap = compute_vwap(sample_ohlcv)
        assert len(vwap) == len(sample_ohlcv)

    def test_vwap_within_price_range(self, sample_ohlcv):
        """VWAP should be between the overall low and high of the data."""
        vwap = compute_vwap(sample_ohlcv)
        # Each VWAP value should be within the cumulative range
        assert vwap.iloc[-1] >= sample_ohlcv["low"].min() - 1
        assert vwap.iloc[-1] <= sample_ohlcv["high"].max() + 1


class TestFullPreprocess:
    """Integration tests for the full preprocess pipeline."""

    def test_preprocess_adds_all_columns(self, sample_ohlcv):
        """Preprocessing should add all expected columns."""
        result = preprocess(sample_ohlcv)
        expected_cols = [
            "volume_delta",
            "cumulative_delta",
            "relative_volume",
            "vwap",
            "support_1",
            "resistance_1",
            "support_2",
            "resistance_2",
            "nearest_sr_distance",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_preprocess_preserves_original_columns(self, sample_ohlcv):
        """Preprocessing should keep original OHLCV columns."""
        result = preprocess(sample_ohlcv)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_preprocess_no_nans_in_key_columns(self, sample_ohlcv):
        """Key computed columns should not have NaN values."""
        result = preprocess(sample_ohlcv)
        # These should have no NaNs
        assert not result["volume_delta"].isna().any()
        assert not result["cumulative_delta"].isna().any()
        assert not result["relative_volume"].isna().any()
        assert not result["vwap"].isna().any()
