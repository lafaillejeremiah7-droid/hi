"""Larry Williams-style daily-bar strategy.

Composed of the four components in :mod:`src.indicators.williams`, each of
which can be enabled on its own so it can be measured standalone before
anything is combined:

- ``gsv``   Greatest Swing Value volatility breakout (the core)
- ``oops``  the Oops! gap-reversal pattern
- ``smash`` the Smash Day reversal
- ``tdom``  trading-day-of-month seasonality

``tdom`` is usable two ways, matching Williams' own use of seasonality as a
bias rather than a system: as an entry component (buy or sell the open on days
whose fitted bias is positive or negative) or as a filter on the other
components (``tdom_filter=True``). Its lookup table must be fitted on train
data and passed in through ``tdom_bias``; the strategy never fits it itself,
so nothing inside the strategy can see beyond the train period.

Exits are fixed-point only, plus a day-count exit and an optional trailing stop
on daily closes. No partial closes, no staged advancement, no probability
models. The daily engine reads ``stop_points``, ``target_points``,
``max_hold_days`` and ``trailing_stop_points`` from ``params``.

When several components arm a buy stop on the same day the lowest one is used,
because that is the resting order that would fill first.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.indicators.williams import (
    gsv_triggers,
    oops_triggers,
    smash_day_triggers,
    tdom_bias_flags,
)
from src.strategies.base import BaseStrategy

ALL_COMPONENTS = ("gsv", "oops", "smash", "tdom")


class WilliamsStrategy(BaseStrategy):
    """Daily stop-entry strategy built from Williams' components."""

    def default_params(self) -> dict[str, Any]:
        """Return default parameters.

        Returns:
            Default parameter dict.
        """
        return {
            # Which entry components are active.
            "components": ("gsv",),
            # GSV
            "gsv_lookback": 5,
            "gsv_multiplier": 0.8,
            "gsv_inverted": False,
            # Smash Day
            "smash_lookback": 5,
            # TDOM: table fitted on train, {trading_day_index: mean points}.
            "tdom_bias": None,
            "tdom_filter": False,
            # Exits
            "stop_points": 50,
            "target_points": 100,
            "max_hold_days": 3,
            "trailing_stop_points": None,
        }

    def _component_triggers(self, df: pd.DataFrame, component: str) -> pd.DataFrame:
        """Entry stop levels for one component.

        Args:
            df: Daily bars.
            component: One of ALL_COMPONENTS.

        Returns:
            DataFrame with long_trigger and short_trigger.
        """
        if component == "gsv":
            return gsv_triggers(
                df,
                lookback=int(self.params["gsv_lookback"]),
                multiplier=float(self.params["gsv_multiplier"]),
                inverted=bool(self.params["gsv_inverted"]),
            )
        if component == "oops":
            return oops_triggers(df)
        if component == "smash":
            return smash_day_triggers(df, lookback=int(self.params["smash_lookback"]))
        if component == "tdom":
            table = self.params.get("tdom_bias") or {}
            bias = tdom_bias_flags(df.index, table)
            # A bias flag is a market order at the open, expressed as a stop
            # order sitting at the open so the engine fills it there.
            return pd.DataFrame(
                {
                    "long_trigger": df["open"].where(bias > 0),
                    "short_trigger": df["open"].where(bias < 0),
                },
                index=df.index,
            )
        raise ValueError(f"Unknown Williams component: {component}")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the daily entry stop levels.

        Args:
            df: Daily RTH bars with open/high/low/close/prior_close.

        Returns:
            DataFrame with:
            - ``long_trigger`` / ``short_trigger``: resting stop-order prices,
              NaN when that side is not armed. Every value is computable at the
              session open.
            - ``component``: which component supplied the armed trigger.
            - ``signal``: 1 when a long is armed, -1 when only a short is armed,
              0 otherwise. Present for BaseStrategy compatibility; the daily
              engine uses the trigger columns, which also carry the case where
              both sides are armed.
        """
        components = tuple(self.params["components"])
        index = df.index

        long_trigger = pd.Series(np.nan, index=index)
        short_trigger = pd.Series(np.nan, index=index)
        long_source = pd.Series("", index=index, dtype=object)
        short_source = pd.Series("", index=index, dtype=object)

        for component in components:
            triggers = self._component_triggers(df, component)

            # The lowest armed buy stop is the one that fills first.
            better_long = triggers["long_trigger"].notna() & (
                long_trigger.isna() | (triggers["long_trigger"] < long_trigger)
            )
            long_trigger = long_trigger.where(~better_long, triggers["long_trigger"])
            long_source = long_source.where(~better_long, component)

            # The highest armed sell stop fills first.
            better_short = triggers["short_trigger"].notna() & (
                short_trigger.isna() | (triggers["short_trigger"] > short_trigger)
            )
            short_trigger = short_trigger.where(~better_short, triggers["short_trigger"])
            short_source = short_source.where(~better_short, component)

        if self.params.get("tdom_filter"):
            table = self.params.get("tdom_bias") or {}
            bias = tdom_bias_flags(index, table)
            long_trigger = long_trigger.where(bias > 0)
            short_trigger = short_trigger.where(bias < 0)

        signal = pd.Series(0, index=index, dtype=int)
        signal = signal.mask(short_trigger.notna(), -1)
        signal = signal.mask(long_trigger.notna(), 1)

        component = long_source.where(long_trigger.notna(), short_source)

        return pd.DataFrame(
            {
                "signal": signal,
                "long_trigger": long_trigger,
                "short_trigger": short_trigger,
                "component": component,
            },
            index=index,
        )

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Fixed-point stop measured from the bar's close.

        The daily engine measures the stop from the actual stop-order fill
        price instead, using ``params['stop_points']``. This method exists for
        BaseStrategy compatibility.

        Args:
            df: Daily bars.
            idx: Positional index of the entry bar.
            direction: 1 long, -1 short.

        Returns:
            Stop price.
        """
        return float(df["close"].iloc[idx]) - float(self.params["stop_points"]) * direction

    def get_take_profit(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: int,
        feature_zscore: float | None = None,
    ) -> float:
        """Fixed-point target measured from the bar's close.

        Args:
            df: Daily bars.
            idx: Positional index of the entry bar.
            direction: 1 long, -1 short.
            feature_zscore: Unused; Williams has no volatility-conditional target.

        Returns:
            Target price.
        """
        return float(df["close"].iloc[idx]) + float(self.params["target_points"]) * direction

    def get_param_ranges(self) -> dict[str, list]:
        """Return the tunable surface.

        Kept deliberately small: Williams' argument is that a system with few
        parameters is less likely to be curve-fitted.

        Returns:
            Mapping of parameter name -> candidate values.
        """
        return {
            "gsv_lookback": [3, 5, 10],
            "gsv_multiplier": [0.6, 1.0],
            "stop_points": [50, 75],
            "max_hold_days": [1, 3, 5],
        }
