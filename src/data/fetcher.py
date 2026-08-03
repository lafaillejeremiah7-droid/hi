"""Data fetching module.

Loads NQ futures 5-minute data with real order flow from aggregated
Databento trade files (primary), can re-fetch from Databento API if
file is missing, and falls back to yfinance for environments without data.

Data source priority:
1. Real order flow parquet (data/NQ_5min_real_orderflow.parquet)
2. Run trade aggregation from raw trade files (data/trades/)
3. Legacy OHLCV parquet (data/NQ_5min_2021_2026.parquet)
4. Databento API fetch
5. yfinance fallback

Returns DataFrames with columns: open, high, low, close, volume
(plus bid_volume, ask_volume, delta, trade_count, avg_trade_size
when loaded from real order flow source) and a DatetimeIndex in
US/Eastern timezone for session filtering.
"""

from pathlib import Path

import pandas as pd

from src.config import get_data_config, get_project_root


def fetch_data(config: dict) -> pd.DataFrame:
    """Fetch NQ futures data based on configuration.

    Data source priority:
    1. Load real order flow parquet (aggregated from Databento trades)
    2. Aggregate from raw trade files if available
    3. Load legacy OHLCV parquet
    4. Re-fetch from Databento API
    5. Fall back to yfinance

    Args:
        config: Full configuration dictionary.

    Returns:
        DataFrame with OHLCV data (+ order flow columns if available)
        indexed by DatetimeIndex in US/Eastern timezone.

    Raises:
        RuntimeError: If no data can be loaded from any source.
    """
    data_config = get_data_config(config)
    source = data_config.get("source", "real_orderflow")
    project_root = get_project_root()

    # Priority 1: Real order flow parquet
    real_of_file = data_config.get("data_file", "data/NQ_5min_real_orderflow.parquet")
    real_of_path = project_root / real_of_file
    if real_of_path.exists():
        df = _load_real_orderflow_parquet(real_of_path)
        if df is not None and len(df) > 0:
            print(f"[Data] Loaded {len(df):,} bars from {real_of_file} (real order flow)")
            return df

    # Priority 2: Aggregate from raw trade files
    trades_dir = project_root / "data" / "trades"
    if trades_dir.exists() and any(trades_dir.glob("NQ_trades_*.parquet")):
        try:
            from src.data.trade_aggregator import load_or_aggregate_bars
            df = load_or_aggregate_bars(
                trades_dir=str(trades_dir),
                output_file=str(real_of_path),
                verbose=True,
            )
            if df is not None and len(df) > 0:
                print(f"[Data] Aggregated {len(df):,} bars from raw trade files")
                return df
        except Exception as e:
            print(f"[Data] Trade aggregation failed: {e}")

    # Priority 3: Legacy OHLCV parquet
    legacy_file = data_config.get("legacy_data_file", "data/NQ_5min_2021_2026.parquet")
    legacy_path = project_root / legacy_file
    if legacy_path.exists():
        df = _load_parquet(legacy_path)
        if df is not None and len(df) > 0:
            print(f"[Data] Loaded {len(df):,} bars from {legacy_file} (legacy OHLCV)")
            return df

    # Priority 4: Databento API
    if source in ("databento", "real_orderflow"):
        db_config = data_config.get("databento", {})
        api_key = db_config.get("api_key", "")
        if api_key:
            df = _fetch_databento(db_config, legacy_path)
            if df is not None and len(df) > 0:
                print(f"[Data] Fetched {len(df):,} bars from Databento API")
                return df
            print("[Data] Databento fetch failed, trying yfinance fallback")

    # Priority 5: yfinance fallback
    yf_config = data_config.get("yfinance", {})
    df = _fetch_yfinance(yf_config, project_root / data_config.get("cache_dir", "data"))
    if df is not None and len(df) > 0:
        print(f"[Data] Fetched {len(df):,} bars from yfinance (fallback)")
        return df

    raise RuntimeError(
        "Failed to load data from any source. "
        "Ensure data/NQ_5min_real_orderflow.parquet exists or raw trade files "
        "are in data/trades/, or provide a valid Databento API key."
    )


def _load_real_orderflow_parquet(path: Path) -> pd.DataFrame | None:
    """Load real order flow parquet file.

    Expects columns: open, high, low, close, volume, bid_volume,
    ask_volume, delta, trade_count, avg_trade_size.

    Args:
        path: Path to real order flow parquet file.

    Returns:
        DataFrame or None on failure.
    """
    try:
        df = pd.read_parquet(path)

        if df.empty:
            return None

        # Ensure timezone is US/Eastern
        if hasattr(df.index, "tz") and df.index.tz is not None:
            if str(df.index.tz) != "US/Eastern":
                df.index = df.index.tz_convert("US/Eastern")
        elif hasattr(df.index, "tz"):
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")

        # Verify essential columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                print(f"[Data] Warning: missing column {col} in {path}")
                return None

        # Drop NaN rows in OHLCV
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[df["volume"] > 0]

        return df

    except Exception as e:
        print(f"[Data] Error loading {path}: {e}")
        return None


def _load_parquet(path: Path) -> pd.DataFrame | None:
    """Load and standardize a parquet data file.

    Handles both capitalized (Open, High, Low, Close, Volume) and
    lowercase column formats. Converts timezone to US/Eastern.

    Args:
        path: Path to parquet file.

    Returns:
        Standardized DataFrame or None on failure.
    """
    try:
        df = pd.read_parquet(path)

        if df.empty:
            return None

        # Standardize column names to lowercase
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower
        if col_map:
            df = df.rename(columns=col_map)

        # Keep only OHLCV columns
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in ohlcv_cols if c in df.columns]
        if len(available_cols) < 5:
            print(f"[Data] Warning: Only found columns {available_cols} in {path}")
            return None
        df = df[available_cols]

        # Handle timezone conversion to US/Eastern
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("US/Eastern")
        else:
            try:
                df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
            except (TypeError, ValueError):
                pass

        # Drop NaN rows and zero volume
        df = df.dropna()
        df = df[df["volume"] > 0]

        return df

    except Exception as e:
        print(f"[Data] Error loading {path}: {e}")
        return None


def _fetch_databento(db_config: dict, save_path: Path) -> pd.DataFrame | None:
    """Fetch data from Databento API and save to parquet.

    Args:
        db_config: Databento configuration section.
        save_path: Path to save the resulting parquet file.

    Returns:
        Standardized DataFrame or None on failure.
    """
    try:
        import databento as db
        from datetime import date

        api_key = db_config.get("api_key", "")
        dataset = db_config.get("dataset", "GLBX.MDP3")
        symbol = db_config.get("symbol", "NQ.c.0")
        schema = db_config.get("schema", "ohlcv-1m")
        resample = db_config.get("resample", "5min")
        start_str = db_config.get("start", "2021-01-01")
        end_str = db_config.get("end", "2026-08-01")
        stype_in = db_config.get("stype_in", "continuous")

        print(f"[Data] Fetching from Databento: {dataset}/{symbol} ({schema})")
        print(f"[Data]   Range: {start_str} to {end_str}")

        client = db.Historical(api_key)
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema=schema,
            start=date.fromisoformat(start_str),
            end=date.fromisoformat(end_str),
            stype_in=stype_in,
        )

        df = data.to_df()

        if df.empty:
            return None

        # Standardize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower
        if col_map:
            df = df.rename(columns=col_map)

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in ohlcv_cols if c in df.columns]
        df = df[available_cols]

        # Resample to target frequency
        if resample and schema == "ohlcv-1m" and resample != "1min":
            df = df.resample(resample).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        if hasattr(df.index, "tz") and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("US/Eastern")

        df = df[df["volume"] > 0]

        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        print(f"[Data] Saved {len(df):,} bars to {save_path}")

        return df

    except ImportError:
        print("[Data] databento package not installed, skipping Databento fetch")
        return None
    except Exception as e:
        print(f"[Data] Databento fetch error: {e}")
        return None


def _fetch_yfinance(yf_config: dict, cache_dir: Path) -> pd.DataFrame | None:
    """Fetch data from yfinance as a fallback.

    Args:
        yf_config: yfinance configuration section.
        cache_dir: Directory for caching parquet files.

    Returns:
        Standardized DataFrame or None on failure.
    """
    try:
        import yfinance as yf

        symbol = yf_config.get("symbol", "NQ=F")
        fallback_symbol = yf_config.get("fallback_symbol", "^NDX")
        period = yf_config.get("period", "730d")
        interval = yf_config.get("interval", "60m")

        cache_dir.mkdir(parents=True, exist_ok=True)

        for sym in [symbol, fallback_symbol]:
            cache_path = cache_dir / f"{sym.replace('=', '_').replace('^', '')}_{period}_{interval}.parquet"

            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    if len(df) > 0:
                        print(f"[Data] Using cached yfinance data from {cache_path}")
                        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                        ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                        df = df[ohlcv]
                        if hasattr(df.index, "tz") and df.index.tz is not None:
                            df.index = df.index.tz_convert("US/Eastern")
                        return df
                except Exception:
                    pass

            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period=period, interval=interval)

                if df is None or df.empty:
                    continue

                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                ohlcv_cols = ["open", "high", "low", "close", "volume"]
                available_cols = [c for c in ohlcv_cols if c in df.columns]
                df = df[available_cols]
                df = df.dropna()
                df = df[df["volume"] > 0]

                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_convert("US/Eastern")
                elif interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"):
                    try:
                        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
                    except (TypeError, ValueError):
                        pass

                df.to_parquet(cache_path)
                print(f"[Data] Fetched {len(df):,} bars for {sym} ({interval}, {period})")
                return df

            except Exception as e:
                print(f"[Data] yfinance error for {sym}: {e}")
                continue

        return None

    except ImportError:
        print("[Data] yfinance package not installed")
        return None
    except Exception as e:
        print(f"[Data] yfinance error: {e}")
        return None
