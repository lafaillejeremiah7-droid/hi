"""Broker data import module.

Imports historical data from various broker export formats and
standardizes them for use in the backtesting framework.

Supported formats:
- Generic CSV
- NinjaTrader (semicolon-separated)
- MetaTrader (comma-separated with specific header)
- cTrader (comma-separated)
- TradingView (comma-separated with 'time' column)
- ThinkOrSwim (tab or comma-separated)

Usage:
    uv run python -m src.data.importer --source ninjatrader --file path/to/export.csv --output data/custom_data.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class DataImporter:
    """Import historical data from various broker export formats."""

    @staticmethod
    def from_csv(
        filepath: str | Path,
        datetime_col: str = "Date",
        datetime_format: str | None = None,
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: str = "Volume",
        timezone: str = "US/Eastern",
        separator: str = ",",
    ) -> pd.DataFrame:
        """Import from generic CSV file.

        Args:
            filepath: Path to the CSV file.
            datetime_col: Name of the datetime column.
            datetime_format: strftime format string for parsing dates.
            open_col: Name of the Open column.
            high_col: Name of the High column.
            low_col: Name of the Low column.
            close_col: Name of the Close column.
            volume_col: Name of the Volume column.
            timezone: Target timezone for the data.
            separator: Column separator character.

        Returns:
            Standardized DataFrame with OHLCV data and DatetimeIndex.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, sep=separator)

        # Parse datetime
        if datetime_col not in df.columns:
            raise ValueError(
                f"Datetime column '{datetime_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if datetime_format:
            df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
        else:
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        df = df.set_index(datetime_col)

        # Map columns
        col_map = {
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            volume_col: "volume",
        }

        missing = [c for c in col_map if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns: {missing}. Available: {list(df.columns)}"
            )

        df = df.rename(columns=col_map)[["open", "high", "low", "close", "volume"]]

        # Handle timezone
        df = _apply_timezone(df, timezone)

        # Validate and clean
        df = _validate_ohlcv(df)

        _print_summary(df, "Generic CSV", filepath)
        return df

    @staticmethod
    def from_ninjatrader(filepath: str | Path, timezone: str = "US/Eastern") -> pd.DataFrame:
        """Import NinjaTrader exported data.

        NinjaTrader exports in semicolon-separated format:
        Date;Time;Open;High;Low;Close;Volume
        or with combined datetime column.

        Args:
            filepath: Path to the NinjaTrader export file.
            timezone: Target timezone (NinjaTrader typically exports in exchange tz).

        Returns:
            Standardized DataFrame with OHLCV data and DatetimeIndex.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Try semicolon separator first (most common NinjaTrader format)
        try:
            df = pd.read_csv(filepath, sep=";")
        except Exception:
            df = pd.read_csv(filepath)

        # NinjaTrader can have Date;Time or combined DateTime column
        if "Date" in df.columns and "Time" in df.columns:
            combined = df["Date"].astype(str) + " " + df["Time"].astype(str)
            # Try NinjaTrader compact format (YYYYMMDD HHMMSS) first
            try:
                df["Datetime"] = pd.to_datetime(combined, format="%Y%m%d %H%M%S")
            except (ValueError, TypeError):
                df["Datetime"] = pd.to_datetime(combined)
        elif "Date" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Date"])
        elif "DateTime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["DateTime"])
        else:
            # Try first column as datetime
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0])

        df = df.set_index("Datetime")

        # Map columns (NinjaTrader uses standard names)
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower in ("close", "last"):
                col_map[col] = "close"
            elif lower in ("volume", "vol"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        ohlcv = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv if c in df.columns]
        df = df[available]

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = _apply_timezone(df, timezone)
        df = _validate_ohlcv(df)

        _print_summary(df, "NinjaTrader", filepath)
        return df

    @staticmethod
    def from_metatrader(filepath: str | Path, timezone: str = "US/Eastern") -> pd.DataFrame:
        """Import MetaTrader exported data.

        MetaTrader (MT4/MT5) exports CSV with format:
        Date,Time,Open,High,Low,Close,Volume
        Date format: YYYY.MM.DD, Time: HH:MM

        Args:
            filepath: Path to the MetaTrader export file.
            timezone: Target timezone (MT typically exports in broker/server tz).

        Returns:
            Standardized DataFrame with OHLCV data and DatetimeIndex.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Try tab separator (MT4 default) then comma
        try:
            df = pd.read_csv(filepath, sep="\t")
            if len(df.columns) < 5:
                df = pd.read_csv(filepath, sep=",")
        except Exception:
            df = pd.read_csv(filepath, sep=",")

        # MetaTrader has Date and Time columns or a combined datetime
        if "Date" in df.columns and "Time" in df.columns:
            df["Datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str)
            )
        elif "<DATE>" in df.columns and "<TIME>" in df.columns:
            # MT4 history center format
            df["Datetime"] = pd.to_datetime(
                df["<DATE>"].astype(str) + " " + df["<TIME>"].astype(str)
            )
        elif "date" in [c.lower() for c in df.columns]:
            date_col = [c for c in df.columns if c.lower() == "date"][0]
            df["Datetime"] = pd.to_datetime(df[date_col])
        else:
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0])

        df = df.set_index("Datetime")

        # Map columns
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip().replace("<", "").replace(">", "")
            if lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower == "close":
                col_map[col] = "close"
            elif lower in ("volume", "vol", "tickvol"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        ohlcv = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv if c in df.columns]
        df = df[available]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = _apply_timezone(df, timezone)
        df = _validate_ohlcv(df)

        _print_summary(df, "MetaTrader", filepath)
        return df

    @staticmethod
    def from_ctrader(filepath: str | Path, timezone: str = "US/Eastern") -> pd.DataFrame:
        """Import cTrader exported data.

        cTrader exports CSV with format:
        Date/Time,Open,High,Low,Close,Volume

        Args:
            filepath: Path to the cTrader export file.
            timezone: Target timezone.

        Returns:
            Standardized DataFrame with OHLCV data and DatetimeIndex.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # cTrader uses "Date/Time" or "Timestamp" as datetime column
        datetime_col = None
        for col in df.columns:
            lower = col.lower()
            if "date" in lower or "time" in lower or "timestamp" in lower:
                datetime_col = col
                break

        if datetime_col is None:
            datetime_col = df.columns[0]

        df["Datetime"] = pd.to_datetime(df[datetime_col])
        df = df.set_index("Datetime")

        # Map columns
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower == "close":
                col_map[col] = "close"
            elif lower in ("volume", "vol"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        ohlcv = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv if c in df.columns]
        df = df[available]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = _apply_timezone(df, timezone)
        df = _validate_ohlcv(df)

        _print_summary(df, "cTrader", filepath)
        return df

    @staticmethod
    def from_tradingview(filepath: str | Path, timezone: str = "US/Eastern") -> pd.DataFrame:
        """Import TradingView exported data.

        TradingView exports CSV with format:
        time,open,high,low,close,Volume (Unix timestamp or ISO datetime)

        Args:
            filepath: Path to the TradingView export file.
            timezone: Target timezone.

        Returns:
            Standardized DataFrame with OHLCV data and DatetimeIndex.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # TradingView uses 'time' column (Unix timestamp or datetime string)
        time_col = None
        for col in df.columns:
            if col.lower() in ("time", "datetime", "date", "timestamp"):
                time_col = col
                break

        if time_col is None:
            time_col = df.columns[0]

        # Try Unix timestamp first, then datetime string
        try:
            # Check if values are numeric (Unix timestamps)
            if pd.to_numeric(df[time_col], errors="coerce").notna().all():
                df["Datetime"] = pd.to_datetime(df[time_col], unit="s", utc=True)
            else:
                df["Datetime"] = pd.to_datetime(df[time_col])
        except (ValueError, TypeError):
            df["Datetime"] = pd.to_datetime(df[time_col])

        df = df.set_index("Datetime")

        # Map columns (TradingView uses lowercase)
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower == "close":
                col_map[col] = "close"
            elif lower in ("volume", "vol"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        ohlcv = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv if c in df.columns]
        df = df[available]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = _apply_timezone(df, timezone)
        df = _validate_ohlcv(df)

        _print_summary(df, "TradingView", filepath)
        return df

    @staticmethod
    def from_thinkorswim(filepath: str | Path, timezone: str = "US/Eastern") -> pd.DataFrame:
        """Import TD Ameritrade/Schwab ThinkOrSwim exported data.

        ThinkOrSwim exports with tab or comma separation:
        DateTime,Open,High,Low,Close,Volume

        Args:
            filepath: Path to the ThinkOrSwim export file.
            timezone: Target timezone (ToS exports in US/Eastern by default).

        Returns:
            Standardized DataFrame with OHLCV data and DatetimeIndex.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # ThinkOrSwim can export with tabs or commas
        try:
            df = pd.read_csv(filepath, sep=",")
            if len(df.columns) < 5:
                df = pd.read_csv(filepath, sep="\t")
        except Exception:
            df = pd.read_csv(filepath, sep="\t")

        # Find datetime column
        datetime_col = None
        for col in df.columns:
            lower = col.lower().strip()
            if "datetime" in lower or "date" in lower or "time" in lower:
                datetime_col = col
                break

        if datetime_col is None:
            datetime_col = df.columns[0]

        df["Datetime"] = pd.to_datetime(df[datetime_col])
        df = df.set_index("Datetime")

        # Map columns
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower in ("close", "last"):
                col_map[col] = "close"
            elif lower in ("volume", "vol"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        ohlcv = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv if c in df.columns]
        df = df[available]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = _apply_timezone(df, timezone)
        df = _validate_ohlcv(df)

        _print_summary(df, "ThinkOrSwim", filepath)
        return df


def _apply_timezone(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Apply timezone conversion to the DataFrame index.

    Args:
        df: DataFrame with DatetimeIndex.
        timezone: Target timezone string.

    Returns:
        DataFrame with timezone-converted index.
    """
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert(timezone)
    else:
        try:
            df.index = df.index.tz_localize(timezone)
        except (TypeError, ValueError):
            # Already localized or ambiguous
            pass
    return df


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLCV data.

    Checks:
    - No NaN values in OHLCV
    - High >= max(Open, Close) for each bar
    - Low <= min(Open, Close) for each bar
    - Volume >= 0
    - Data sorted chronologically

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        Cleaned DataFrame with invalid rows removed.
    """
    initial_len = len(df)

    # Drop NaN rows
    df = df.dropna()

    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Validate OHLC relationships
    valid_high = df["high"] >= df[["open", "close"]].max(axis=1) - 1e-8
    valid_low = df["low"] <= df[["open", "close"]].min(axis=1) + 1e-8
    valid_volume = df["volume"] >= 0
    valid_mask = valid_high & valid_low & valid_volume

    df = df[valid_mask]

    # Sort by index
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[Data] Dropped {dropped} invalid/duplicate rows during validation")

    return df


def _print_summary(df: pd.DataFrame, source: str, filepath: Path) -> None:
    """Print import summary.

    Args:
        df: Imported DataFrame.
        source: Source name for display.
        filepath: Original file path.
    """
    if len(df) == 0:
        print(f"[Import] {source}: No valid data imported from {filepath.name}")
        return

    # Estimate timeframe
    if len(df) > 1:
        avg_diff = (df.index[-1] - df.index[0]) / (len(df) - 1)
        if avg_diff.total_seconds() < 120:
            timeframe = "~1 min"
        elif avg_diff.total_seconds() < 600:
            timeframe = "~5 min"
        elif avg_diff.total_seconds() < 3600:
            timeframe = f"~{int(avg_diff.total_seconds() / 60)} min"
        elif avg_diff.total_seconds() < 86400:
            timeframe = f"~{int(avg_diff.total_seconds() / 3600)} hour"
        else:
            timeframe = "~daily"
    else:
        timeframe = "unknown"

    print(f"[Import] {source}: {len(df):,} bars ({timeframe})")
    print(f"[Import]   File: {filepath.name}")
    print(f"[Import]   Range: {df.index[0]} to {df.index[-1]}")
    print(f"[Import]   Columns: {list(df.columns)}")


def main():
    """CLI entry point for data import."""
    parser = argparse.ArgumentParser(
        description="Import broker data into the backtesting framework"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["csv", "ninjatrader", "metatrader", "ctrader", "tradingview", "thinkorswim"],
        help="Broker/source format",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input data file",
    )
    parser.add_argument(
        "--output",
        default="data/custom_data.parquet",
        help="Output parquet file path (default: data/custom_data.parquet)",
    )
    parser.add_argument(
        "--timezone",
        default="US/Eastern",
        help="Target timezone (default: US/Eastern)",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="Column separator for CSV format (default: ',')",
    )
    parser.add_argument(
        "--datetime-col",
        default="Date",
        help="Datetime column name for CSV format (default: 'Date')",
    )
    parser.add_argument(
        "--datetime-format",
        default=None,
        help="Datetime format string for CSV format (default: auto-detect)",
    )

    args = parser.parse_args()

    importer = DataImporter()
    source_map = {
        "csv": lambda: importer.from_csv(
            args.file,
            datetime_col=args.datetime_col,
            datetime_format=args.datetime_format,
            timezone=args.timezone,
            separator=args.separator,
        ),
        "ninjatrader": lambda: importer.from_ninjatrader(args.file, timezone=args.timezone),
        "metatrader": lambda: importer.from_metatrader(args.file, timezone=args.timezone),
        "ctrader": lambda: importer.from_ctrader(args.file, timezone=args.timezone),
        "tradingview": lambda: importer.from_tradingview(args.file, timezone=args.timezone),
        "thinkorswim": lambda: importer.from_thinkorswim(args.file, timezone=args.timezone),
    }

    try:
        df = source_map[args.source]()

        # Save to parquet
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        print(f"\n[Import] Saved {len(df):,} bars to {output_path}")
        print(f"[Import] Ready for backtesting!")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
