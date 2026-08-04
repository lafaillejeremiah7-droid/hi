"""Daily RTH bar construction from 1-minute NQ data.

Larry Williams' systems are daily-bar systems. The 1-minute file is the only
clean long-history source in this project, so the daily bars are built from it
rather than downloaded again.

One bar per trading day, built from the 09:30-16:00 ET regular session:

- open   first 1-min Open at or after 09:30 ET
- high   session high
- low    session low
- close  last 1-min Close before 16:00 ET
- volume session volume
- prior_close  the previous session's close, so the overnight gap
  (open - prior_close) is measurable

Two extra columns, ``high_time`` and ``low_time``, record the minute at which
the session high and low occurred. They are not signal inputs. They are used
by the daily simulator to resolve intraday ordering (which of two resting stop
orders filled first, whether a stop or a target was reached first) instead of
guessing. Without them every ambiguous day would have to be resolved
pessimistically.
"""

from pathlib import Path

import pandas as pd

RTH_START = "09:30"
RTH_END = "16:00"
EASTERN = "US/Eastern"

# A regular session is 390 minutes; CME early closes are 210. Anything much
# shorter is a data gap, not a trading day.
MIN_SESSION_BARS = 120

DAILY_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "prior_close",
    "high_time",
    "low_time",
    "n_bars",
]


def _minutes(hhmm: str) -> int:
    """Convert an HH:MM string to minutes since midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def resample_to_daily_rth(
    minute_df: pd.DataFrame,
    session_start: str = RTH_START,
    session_end: str = RTH_END,
    timezone: str = EASTERN,
    min_session_bars: int = MIN_SESSION_BARS,
) -> pd.DataFrame:
    """Resample 1-minute bars into daily regular-session bars.

    Args:
        minute_df: 1-minute OHLCV frame with a tz-aware DatetimeIndex. Column
            names may be capitalized (Open/High/Low/Close/Volume) or lower case.
        session_start: Session start in the target timezone, HH:MM.
        session_end: Session end in the target timezone, HH:MM (exclusive).
        timezone: Timezone the session is defined in.
        min_session_bars: Days with fewer 1-min bars than this are dropped as
            data gaps rather than treated as trading days.

    Returns:
        DataFrame indexed by the session open timestamp (tz-aware, timezone),
        with the columns in DAILY_COLUMNS.
    """
    df = minute_df.rename(columns={c: c.lower() for c in minute_df.columns})

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"1-minute data is missing columns: {sorted(missing)}")

    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert(timezone).sort_index()

    bar_minutes = df.index.hour * 60 + df.index.minute
    in_rth = (bar_minutes >= _minutes(session_start)) & (bar_minutes < _minutes(session_end))
    rth = df.loc[in_rth]

    if rth.empty:
        raise ValueError("No 1-minute bars fall inside the requested session")

    grouped = rth.groupby(rth.index.normalize(), sort=True)

    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "high_time": grouped["high"].idxmax(),
            "low_time": grouped["low"].idxmin(),
            "n_bars": grouped["close"].size(),
        }
    )

    daily = daily.loc[daily["n_bars"] >= min_session_bars]
    daily.index = daily.index + pd.Timedelta(minutes=_minutes(session_start))
    daily.index.name = "Date"

    # The previous session's close. Shift on the already-sorted frame so the
    # value at day i is only ever from day i-1.
    daily["prior_close"] = daily["close"].shift(1)

    return daily[DAILY_COLUMNS]


def load_daily_rth_bars(
    minute_path: str | Path,
    cache_path: str | Path,
    session_start: str = RTH_START,
    session_end: str = RTH_END,
    timezone: str = EASTERN,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load daily RTH bars, building and caching them from 1-minute data.

    Args:
        minute_path: Path to the 1-minute parquet file.
        cache_path: Path the daily bars are cached to.
        session_start: Session start HH:MM.
        session_end: Session end HH:MM.
        timezone: Session timezone.
        force_rebuild: Rebuild even if the cache exists.

    Returns:
        Daily RTH bar DataFrame.
    """
    cache = Path(cache_path)
    if cache.exists() and not force_rebuild:
        cached = pd.read_parquet(cache)
        if set(DAILY_COLUMNS).issubset(cached.columns):
            return cached[DAILY_COLUMNS]

    minute_df = pd.read_parquet(minute_path)
    daily = resample_to_daily_rth(
        minute_df,
        session_start=session_start,
        session_end=session_end,
        timezone=timezone,
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache)
    return daily


def overnight_gap_points(daily: pd.DataFrame) -> pd.Series:
    """Overnight gap in points: today's RTH open minus the prior RTH close.

    Args:
        daily: Daily RTH bars.

    Returns:
        Series of signed gaps in index points.
    """
    return daily["open"] - daily["prior_close"]
