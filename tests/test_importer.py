"""Tests for the DataImporter and Databento data loading.

Tests each broker format parser, data validation, and the
real 5-min parquet file.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.importer import DataImporter, _validate_ohlcv, _apply_timezone


# --- Fixtures ---


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    data = pd.DataFrame({
        "Date": pd.date_range("2024-01-02 09:30", periods=10, freq="5min"),
        "Open": np.random.uniform(15000, 15100, 10),
        "High": np.random.uniform(15100, 15200, 10),
        "Low": np.random.uniform(14900, 15000, 10),
        "Close": np.random.uniform(15000, 15100, 10),
        "Volume": np.random.randint(100, 5000, 10),
    })
    # Ensure OHLC relationships
    data["High"] = data[["Open", "High", "Close"]].max(axis=1) + 5
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1) - 5

    filepath = tmp_path / "test_data.csv"
    data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def ninjatrader_csv_file(tmp_path):
    """Create a NinjaTrader-style CSV file."""
    lines = [
        "Date;Time;Open;High;Low;Close;Volume",
        "20240102;093000;15000.50;15010.00;14990.00;15005.25;1234",
        "20240102;093500;15005.25;15015.00;15000.00;15012.50;987",
        "20240102;094000;15012.50;15020.00;15008.00;15018.75;1456",
        "20240102;094500;15018.75;15025.00;15015.00;15022.00;876",
        "20240102;095000;15022.00;15030.00;15020.00;15028.50;1123",
    ]
    filepath = tmp_path / "ninja_data.csv"
    filepath.write_text("\n".join(lines))
    return filepath


@pytest.fixture
def metatrader_csv_file(tmp_path):
    """Create a MetaTrader-style CSV file."""
    lines = [
        "Date,Time,Open,High,Low,Close,Volume",
        "2024.01.02,09:30,15000.50,15010.00,14990.00,15005.25,1234",
        "2024.01.02,09:35,15005.25,15015.00,15000.00,15012.50,987",
        "2024.01.02,09:40,15012.50,15020.00,15008.00,15018.75,1456",
        "2024.01.02,09:45,15018.75,15025.00,15015.00,15022.00,876",
        "2024.01.02,09:50,15022.00,15030.00,15020.00,15028.50,1123",
    ]
    filepath = tmp_path / "mt_data.csv"
    filepath.write_text("\n".join(lines))
    return filepath


@pytest.fixture
def tradingview_csv_file(tmp_path):
    """Create a TradingView-style CSV file."""
    lines = [
        "time,open,high,low,close,Volume",
        "2024-01-02 09:30:00,15000.50,15010.00,14990.00,15005.25,1234",
        "2024-01-02 09:35:00,15005.25,15015.00,15000.00,15012.50,987",
        "2024-01-02 09:40:00,15012.50,15020.00,15008.00,15018.75,1456",
        "2024-01-02 09:45:00,15018.75,15025.00,15015.00,15022.00,876",
        "2024-01-02 09:50:00,15022.00,15030.00,15020.00,15028.50,1123",
    ]
    filepath = tmp_path / "tv_data.csv"
    filepath.write_text("\n".join(lines))
    return filepath


@pytest.fixture
def ctrader_csv_file(tmp_path):
    """Create a cTrader-style CSV file."""
    lines = [
        "Date/Time,Open,High,Low,Close,Volume",
        "2024-01-02 09:30:00,15000.50,15010.00,14990.00,15005.25,1234",
        "2024-01-02 09:35:00,15005.25,15015.00,15000.00,15012.50,987",
        "2024-01-02 09:40:00,15012.50,15020.00,15008.00,15018.75,1456",
    ]
    filepath = tmp_path / "ct_data.csv"
    filepath.write_text("\n".join(lines))
    return filepath


@pytest.fixture
def thinkorswim_csv_file(tmp_path):
    """Create a ThinkOrSwim-style CSV file."""
    lines = [
        "DateTime,Open,High,Low,Close,Volume",
        "01/02/2024 09:30,15000.50,15010.00,14990.00,15005.25,1234",
        "01/02/2024 09:35,15005.25,15015.00,15000.00,15012.50,987",
        "01/02/2024 09:40,15012.50,15020.00,15008.00,15018.75,1456",
        "01/02/2024 09:45,15018.75,15025.00,15015.00,15022.00,876",
    ]
    filepath = tmp_path / "tos_data.csv"
    filepath.write_text("\n".join(lines))
    return filepath


# --- Generic CSV Import Tests ---


class TestCSVImport:
    """Tests for generic CSV import."""

    def test_imports_csv_with_correct_columns(self, sample_csv_file):
        """Imports CSV and returns correct column structure."""
        df = DataImporter.from_csv(sample_csv_file)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) == 10

    def test_csv_has_datetime_index(self, sample_csv_file):
        """Imported data has DatetimeIndex."""
        df = DataImporter.from_csv(sample_csv_file)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_csv_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            DataImporter.from_csv("/nonexistent/file.csv")

    def test_csv_missing_column(self, tmp_path):
        """Raises ValueError for missing required columns."""
        filepath = tmp_path / "bad.csv"
        filepath.write_text("Date,Open,High\n2024-01-01,100,101\n")
        with pytest.raises(ValueError, match="Missing columns"):
            DataImporter.from_csv(filepath)

    def test_csv_custom_separator(self, tmp_path):
        """Supports custom separators."""
        lines = [
            "Date;Open;High;Low;Close;Volume",
            "2024-01-02 09:30;15000;15010;14990;15005;1234",
            "2024-01-02 09:35;15005;15015;15000;15012;987",
        ]
        filepath = tmp_path / "semi.csv"
        filepath.write_text("\n".join(lines))
        df = DataImporter.from_csv(filepath, separator=";")
        assert len(df) == 2


# --- NinjaTrader Import Tests ---


class TestNinjaTraderImport:
    """Tests for NinjaTrader format import."""

    def test_imports_ninjatrader_data(self, ninjatrader_csv_file):
        """Correctly parses NinjaTrader semicolon-separated format."""
        df = DataImporter.from_ninjatrader(ninjatrader_csv_file)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) == 5

    def test_ninjatrader_values_correct(self, ninjatrader_csv_file):
        """Values are correctly parsed from NinjaTrader format."""
        df = DataImporter.from_ninjatrader(ninjatrader_csv_file)
        assert df["open"].iloc[0] == pytest.approx(15000.50)
        assert df["volume"].iloc[0] == 1234

    def test_ninjatrader_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            DataImporter.from_ninjatrader("/nonexistent/file.csv")


# --- MetaTrader Import Tests ---


class TestMetaTraderImport:
    """Tests for MetaTrader format import."""

    def test_imports_metatrader_data(self, metatrader_csv_file):
        """Correctly parses MetaTrader format."""
        df = DataImporter.from_metatrader(metatrader_csv_file)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) == 5

    def test_metatrader_values_correct(self, metatrader_csv_file):
        """Values are correctly parsed from MetaTrader format."""
        df = DataImporter.from_metatrader(metatrader_csv_file)
        assert df["open"].iloc[0] == pytest.approx(15000.50)
        assert df["high"].iloc[0] == pytest.approx(15010.00)


# --- TradingView Import Tests ---


class TestTradingViewImport:
    """Tests for TradingView format import."""

    def test_imports_tradingview_data(self, tradingview_csv_file):
        """Correctly parses TradingView format."""
        df = DataImporter.from_tradingview(tradingview_csv_file)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) == 5

    def test_tradingview_has_datetime_index(self, tradingview_csv_file):
        """TradingView import has proper DatetimeIndex."""
        df = DataImporter.from_tradingview(tradingview_csv_file)
        assert isinstance(df.index, pd.DatetimeIndex)


# --- cTrader Import Tests ---


class TestCTraderImport:
    """Tests for cTrader format import."""

    def test_imports_ctrader_data(self, ctrader_csv_file):
        """Correctly parses cTrader format."""
        df = DataImporter.from_ctrader(ctrader_csv_file)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) == 3


# --- ThinkOrSwim Import Tests ---


class TestThinkOrSwimImport:
    """Tests for ThinkOrSwim format import."""

    def test_imports_thinkorswim_data(self, thinkorswim_csv_file):
        """Correctly parses ThinkOrSwim format."""
        df = DataImporter.from_thinkorswim(thinkorswim_csv_file)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) == 4

    def test_thinkorswim_values_correct(self, thinkorswim_csv_file):
        """Values are correctly parsed from ThinkOrSwim format."""
        df = DataImporter.from_thinkorswim(thinkorswim_csv_file)
        assert df["close"].iloc[0] == pytest.approx(15005.25)


# --- Data Validation Tests ---


class TestDataValidation:
    """Tests for OHLCV data validation."""

    def test_validates_ohlc_relationship(self):
        """Drops rows where high < close or low > close."""
        df = pd.DataFrame({
            "open": [100, 100, 100],
            "high": [110, 90, 110],  # Row 1: high < open (invalid)
            "low": [90, 90, 95],
            "close": [105, 95, 105],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        result = _validate_ohlcv(df)
        # Row 1 (high=90 < close=95 and open=100) should be dropped
        assert len(result) == 2

    def test_drops_nan_rows(self):
        """Drops rows with NaN values."""
        df = pd.DataFrame({
            "open": [100, np.nan, 100],
            "high": [110, 110, 110],
            "low": [90, 90, 90],
            "close": [105, 105, 105],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        result = _validate_ohlcv(df)
        assert len(result) == 2

    def test_sorts_by_index(self):
        """Sorts data chronologically."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [110, 111, 112],
            "low": [90, 91, 92],
            "close": [105, 106, 107],
            "volume": [1000, 1000, 1000],
        }, index=pd.DatetimeIndex([
            "2024-01-03", "2024-01-01", "2024-01-02"
        ]))

        result = _validate_ohlcv(df)
        assert result.index[0] < result.index[1] < result.index[2]


# --- Databento Parquet Loading Tests ---


class TestDatabentoPArquetLoading:
    """Tests for the real 5-min Databento data file."""

    @pytest.fixture
    def data_path(self):
        """Path to the real 5-min parquet file."""
        return Path("data/NQ_5min_2021_2026.parquet")

    def test_parquet_file_exists(self, data_path):
        """The 5-min data file exists."""
        assert data_path.exists(), f"Expected data file at {data_path}"

    def test_parquet_has_expected_shape(self, data_path):
        """Data file has approximately the expected number of bars."""
        df = pd.read_parquet(data_path)
        # Should have ~393K bars
        assert len(df) > 350000, f"Expected >350K bars, got {len(df)}"
        assert len(df) < 500000, f"Expected <500K bars, got {len(df)}"

    def test_parquet_has_ohlcv_columns(self, data_path):
        """Data file has OHLCV columns."""
        df = pd.read_parquet(data_path)
        # Can be capitalized or lowercase
        cols_lower = {c.lower() for c in df.columns}
        assert "open" in cols_lower
        assert "high" in cols_lower
        assert "low" in cols_lower
        assert "close" in cols_lower
        assert "volume" in cols_lower

    def test_parquet_has_datetime_index(self, data_path):
        """Data file has a DatetimeIndex."""
        df = pd.read_parquet(data_path)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_parquet_date_range(self, data_path):
        """Data covers Jan 2021 to Jul 2026."""
        df = pd.read_parquet(data_path)
        assert df.index[0].year == 2021
        assert df.index[-1].year == 2026

    def test_fetcher_loads_data(self):
        """fetch_data() loads the parquet file correctly."""
        from src.config import load_config
        from src.data.fetcher import fetch_data

        config = load_config()
        df = fetch_data(config)
        # Real orderflow file has ~264K bars (3.7 years), legacy has ~393K
        assert len(df) > 200000
        assert "open" in df.columns
        assert "close" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        # Should be in US/Eastern
        if df.index.tz is not None:
            assert "Eastern" in str(df.index.tz) or "US/Eastern" in str(df.index.tz)

    def test_fetcher_data_is_5min(self):
        """Loaded data has approximately 5-minute bar intervals."""
        from src.config import load_config
        from src.data.fetcher import fetch_data

        config = load_config()
        df = fetch_data(config)
        # Check median time difference (should be ~5 min)
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median().total_seconds()
        assert 250 < median_diff < 400, f"Median bar interval: {median_diff}s (expected ~300s)"
