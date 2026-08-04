"""CFTC Commitments of Traders data and Williams' COT Index.

Williams treated commercial positioning as his primary directional bias and
price patterns as timing. The COT Index he used is a min-max normalization of
the commercial net position over a multi-year lookback:

    COT Index = 100 * (net - min(net, lookback)) / (max(net, lookback) - min(...))

Source: the CFTC legacy futures-only history files, one zip per year, which
contain the commercial long and short columns. The E-mini Nasdaq-100 contract
appears there as "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE".

Publication lag is modelled: a report is dated as of Tuesday but released the
following Friday afternoon, so a daily bar may only use reports whose release
date precedes it.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COT_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

# The CFTC renamed the E-mini Nasdaq-100 contract during 2022. Both names refer
# to the same contract, so both are accepted and de-duplicated by report date.
COT_CONTRACT_NAMES = (
    "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE",
)
COT_CONTRACT = "E-mini Nasdaq-100 (CFTC: NASDAQ MINI / NASDAQ-100 STOCK INDEX (MINI))"

NAME_COLUMN = "Market and Exchange Names"
DATE_COLUMN = "As of Date in Form YYYY-MM-DD"
LONG_COLUMN = "Commercial Positions-Long (All)"
SHORT_COLUMN = "Commercial Positions-Short (All)"

# A report dated Tuesday is released the following Friday.
PUBLICATION_LAG_DAYS = 3


def _download_year(year: int, timeout: int = 60) -> pd.DataFrame:
    """Download and parse one annual legacy COT file.

    Args:
        year: Calendar year.
        timeout: Socket timeout in seconds.

    Returns:
        Rows for COT_CONTRACT with the date, commercial long and short columns.
    """
    import urllib.request

    url = COT_URL.format(year=year)
    request = urllib.request.Request(url, headers={"User-Agent": "nas100-backtest/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            frame = pd.read_csv(
                io.TextIOWrapper(handle, encoding="latin-1"),
                usecols=[NAME_COLUMN, DATE_COLUMN, LONG_COLUMN, SHORT_COLUMN],
                low_memory=False,
            )

    frame[NAME_COLUMN] = frame[NAME_COLUMN].str.strip()
    return frame.loc[frame[NAME_COLUMN].isin(COT_CONTRACT_NAMES)].copy()


def fetch_cot_weekly(start_year: int = 2018, end_year: int = 2026) -> pd.DataFrame:
    """Fetch the weekly commercial net position for the E-mini Nasdaq-100.

    Args:
        start_year: First year to download.
        end_year: Last year to download (inclusive).

    Returns:
        DataFrame indexed by report date with commercial_long, commercial_short
        and commercial_net.

    Raises:
        RuntimeError: If no year could be downloaded and parsed.
    """
    frames = []
    failures = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(_download_year(year))
        except Exception as exc:  # network, zip or schema problem
            failures.append(f"{year}: {type(exc).__name__}")

    if not frames:
        raise RuntimeError(f"No COT year downloaded cleanly ({'; '.join(failures)})")

    combined = pd.concat(frames, ignore_index=True)
    combined["report_date"] = pd.to_datetime(combined[DATE_COLUMN])
    combined = combined.sort_values("report_date").drop_duplicates("report_date")

    result = pd.DataFrame(
        {
            "commercial_long": combined[LONG_COLUMN].astype(float).to_numpy(),
            "commercial_short": combined[SHORT_COLUMN].astype(float).to_numpy(),
        },
        index=pd.DatetimeIndex(combined["report_date"], name="report_date"),
    )
    result["commercial_net"] = result["commercial_long"] - result["commercial_short"]
    return result


def add_cot_index(weekly: pd.DataFrame, lookback_weeks: int = 156) -> pd.DataFrame:
    """Add Williams' COT Index to a weekly commercial net position frame.

    Args:
        weekly: Frame with a commercial_net column, sorted by report date.
        lookback_weeks: Lookback window (156 weeks = 3 years).

    Returns:
        The frame with a cot_index column in 0-100. The value at week i uses
        weeks i-lookback+1..i only.
    """
    net = weekly["commercial_net"]
    lowest = net.rolling(lookback_weeks, min_periods=lookback_weeks).min()
    highest = net.rolling(lookback_weeks, min_periods=lookback_weeks).max()
    span = (highest - lowest).replace(0.0, np.nan)

    result = weekly.copy()
    result["cot_index"] = (net - lowest) / span * 100.0
    return result


def load_cot_index(
    cache_path: str | Path,
    start_year: int = 2018,
    end_year: int = 2026,
    lookback_weeks: int = 156,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load the weekly COT frame with the COT Index, fetching and caching it.

    Args:
        cache_path: Parquet cache path.
        start_year: First year to fetch.
        end_year: Last year to fetch.
        lookback_weeks: COT Index lookback.
        force_refresh: Re-download even if the cache exists.

    Returns:
        Weekly frame with commercial_net and cot_index.
    """
    cache = Path(cache_path)
    if cache.exists() and not force_refresh:
        cached = pd.read_parquet(cache)
        if "cot_index" in cached.columns:
            return cached

    weekly = add_cot_index(fetch_cot_weekly(start_year, end_year), lookback_weeks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(cache)
    return weekly


def cot_index_daily(weekly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Map the weekly COT Index onto daily bars without look-ahead.

    Each daily bar receives the most recent report whose release date (report
    date + PUBLICATION_LAG_DAYS) is strictly earlier than the bar.

    Args:
        weekly: Frame with a cot_index column indexed by report date.
        index: Daily bar index (may be tz-aware).

    Returns:
        Series of COT Index values aligned to ``index``, NaN before coverage
        starts.
    """
    released = weekly.index + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    available = pd.Series(weekly["cot_index"].to_numpy(), index=released).sort_index()

    bar_dates = pd.DatetimeIndex(index.tz_localize(None) if index.tz else index).normalize()
    positions = np.searchsorted(available.index.to_numpy(), bar_dates.to_numpy(), side="left") - 1

    values = np.full(len(bar_dates), np.nan)
    valid = positions >= 0
    values[valid] = available.to_numpy()[positions[valid]]

    return pd.Series(values, index=index, name="cot_index")
