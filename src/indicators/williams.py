"""Larry Williams daily-bar indicators.

Four independent components, each implemented on its own so it can be tested
standalone before anything is combined:

1. Greatest Swing Value (GSV) volatility breakout - the core.
2. Oops! pattern.
3. Smash Day reversal.
4. Trading Day of Month (TDOM) seasonality.

Causality rule for every function in this module: the value returned for day
``i`` uses only bars up to and including day ``i-1``, plus day ``i``'s open
(which is known the instant the session starts). Nothing reads day ``i``'s
high, low or close. The regression tests in tests/test_lookahead.py enforce
this by truncating the frame.

Definitional choices that the literature is not unanimous about are named
explicitly here, because they change the numbers:

- "up-closing day" means ``close > prior_close`` (Williams' "up close day"),
  not ``close > open``.
- The GSV swing measurements follow the task specification: the buy swing is
  ``open - low`` on up-closing days and the sell swing is ``high - open`` on
  down-closing days. Several published versions of GSV take the swings from
  the *opposite* close days (buy swing from down-close days). The alternative
  is available as :func:`greatest_swing_values_inverted` so the difference can
  be measured instead of argued about.
"""

import numpy as np
import pandas as pd


def _rolling_mean_of_qualifying_days(
    swing: pd.Series, lookback: int, index: pd.Index
) -> pd.Series:
    """Average the last ``lookback`` qualifying swings, as known at day i's open.

    ``swing`` holds a value on qualifying days and NaN elsewhere. The rolling
    mean is taken over qualifying days only, forward-filled to every day, then
    shifted one day so that day i sees only completed days up to i-1.

    Args:
        swing: Swing measurement, NaN on non-qualifying days.
        lookback: Number of qualifying days to average.
        index: Index of the full daily frame.

    Returns:
        Series aligned to ``index``.
    """
    qualifying = swing.dropna()
    if qualifying.empty:
        return pd.Series(np.nan, index=index)

    averaged = qualifying.rolling(lookback, min_periods=lookback).mean()
    return averaged.reindex(index).ffill().shift(1)


def greatest_swing_values(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Greatest Swing Values for the volatility breakout.

    Buy swing: on up-closing days, ``open - low``. Averaged over the last
    ``lookback`` such days gives ``gsv_buy``.
    Sell swing: on down-closing days, ``high - open``. Averaged over the last
    ``lookback`` such days gives ``gsv_sell``.

    Args:
        df: Daily bars with open/high/low/close and prior_close.
        lookback: Number of qualifying days to average (Williams' N).

    Returns:
        DataFrame with columns gsv_buy and gsv_sell, both usable at day i's
        open (they contain no information from day i's own high/low/close).
    """
    prior_close = df["prior_close"]
    up_close = df["close"] > prior_close
    down_close = df["close"] < prior_close

    buy_swing = (df["open"] - df["low"]).where(up_close)
    sell_swing = (df["high"] - df["open"]).where(down_close)

    return pd.DataFrame(
        {
            "gsv_buy": _rolling_mean_of_qualifying_days(buy_swing, lookback, df.index),
            "gsv_sell": _rolling_mean_of_qualifying_days(sell_swing, lookback, df.index),
        },
        index=df.index,
    )


def greatest_swing_values_inverted(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """GSV with the swings taken from the opposite close days.

    Buy swing from down-closing days (``high - open``), sell swing from
    up-closing days (``open - low``). This is the formulation used by several
    published descriptions of GSV. Reported as a diagnostic only; it is not
    part of the tuned grid.

    Args:
        df: Daily bars with open/high/low/close and prior_close.
        lookback: Number of qualifying days to average.

    Returns:
        DataFrame with columns gsv_buy and gsv_sell.
    """
    prior_close = df["prior_close"]
    up_close = df["close"] > prior_close
    down_close = df["close"] < prior_close

    buy_swing = (df["high"] - df["open"]).where(down_close)
    sell_swing = (df["open"] - df["low"]).where(up_close)

    return pd.DataFrame(
        {
            "gsv_buy": _rolling_mean_of_qualifying_days(buy_swing, lookback, df.index),
            "gsv_sell": _rolling_mean_of_qualifying_days(sell_swing, lookback, df.index),
        },
        index=df.index,
    )


def gsv_triggers(
    df: pd.DataFrame, lookback: int = 5, multiplier: float = 0.8, inverted: bool = False
) -> pd.DataFrame:
    """Entry stop levels for the GSV volatility breakout.

    Long stop  = today's open + gsv_buy * multiplier
    Short stop = today's open - gsv_sell * multiplier

    Args:
        df: Daily bars.
        lookback: GSV lookback N.
        multiplier: GSV multiplier.
        inverted: Use the opposite-close-day swing definition.

    Returns:
        DataFrame with long_trigger and short_trigger (NaN where not armed).
    """
    gsv = greatest_swing_values_inverted(df, lookback) if inverted else greatest_swing_values(df, lookback)

    long_trigger = df["open"] + gsv["gsv_buy"] * multiplier
    short_trigger = df["open"] - gsv["gsv_sell"] * multiplier

    return pd.DataFrame(
        {"long_trigger": long_trigger, "short_trigger": short_trigger},
        index=df.index,
    )


def oops_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Williams' Oops! pattern entry levels.

    Long: today opens below yesterday's low; buy stop at yesterday's low, so
    the trade is taken only if price trades back up through it.
    Short: today opens above yesterday's high; sell stop at yesterday's high.

    Args:
        df: Daily bars.

    Returns:
        DataFrame with long_trigger and short_trigger (NaN where not armed).
    """
    prior_low = df["low"].shift(1)
    prior_high = df["high"].shift(1)

    long_trigger = prior_low.where(df["open"] < prior_low)
    short_trigger = prior_high.where(df["open"] > prior_high)

    return pd.DataFrame(
        {"long_trigger": long_trigger, "short_trigger": short_trigger},
        index=df.index,
    )


def smash_day_triggers(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Williams' Smash Day reversal entry levels.

    Long: yesterday closed below the lowest low of the ``lookback`` days that
    preceded it (a downside "smash"); buy stop at yesterday's high, so the
    trade is taken only if today trades back above it.
    Short: yesterday closed above the highest high of the preceding
    ``lookback`` days; sell stop at yesterday's low.

    Args:
        df: Daily bars.
        lookback: Days before yesterday used for the range extreme.

    Returns:
        DataFrame with long_trigger and short_trigger (NaN where not armed).
    """
    prior_close = df["close"].shift(1)
    prior_high = df["high"].shift(1)
    prior_low = df["low"].shift(1)

    # Range of the N days that ended the day before yesterday.
    lowest = df["low"].rolling(lookback, min_periods=lookback).min().shift(2)
    highest = df["high"].rolling(lookback, min_periods=lookback).max().shift(2)

    long_trigger = prior_high.where(prior_close < lowest)
    short_trigger = prior_low.where(prior_close > highest)

    return pd.DataFrame(
        {"long_trigger": long_trigger, "short_trigger": short_trigger},
        index=df.index,
    )


def trading_day_of_month(index: pd.DatetimeIndex) -> pd.Series:
    """1-based trading-day-of-month index for each bar.

    The first trading day present in a calendar month is 1, the next is 2, and
    so on. Computed from the index alone, so it is known in advance.

    Args:
        index: DatetimeIndex of daily bars.

    Returns:
        Integer Series aligned to ``index``.
    """
    month_key = pd.Series(index.year * 12 + index.month, index=index)
    return (month_key.groupby(month_key).cumcount() + 1).astype(int)


def tdom_bias_table(
    df: pd.DataFrame, min_observations: int = 8
) -> dict[int, float]:
    """Average open-to-close return by trading day of month.

    Fit this on TRAIN rows only. The value for each trading-day index is the
    mean same-day open-to-close move in points, which is what a day-directional
    bias flag would have to capture.

    Args:
        df: Daily bars for the fitting period only.
        min_observations: Trading-day indices with fewer observations than this
            are left out of the table (too thin to act on).

    Returns:
        Mapping of trading-day-of-month index -> mean open-to-close points.
    """
    tdom = trading_day_of_month(df.index)
    day_return = df["close"] - df["open"]

    grouped = day_return.groupby(tdom)
    means = grouped.mean()
    counts = grouped.size()

    return {
        int(day): float(mean)
        for day, mean in means.items()
        if counts.loc[day] >= min_observations
    }


def tdom_bias_flags(index: pd.DatetimeIndex, table: dict[int, float]) -> pd.Series:
    """Directional bias flag per day from a fitted TDOM table.

    Args:
        index: DatetimeIndex of the days to flag.
        table: Table from :func:`tdom_bias_table`, fitted on train data.

    Returns:
        Series of +1 (long bias), -1 (short bias) or 0 (no bias / not in table).
    """
    tdom = trading_day_of_month(index)
    mapped = tdom.map(lambda day: table.get(int(day), 0.0))
    return np.sign(mapped).astype(int)
