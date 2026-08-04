"""Daily-bar backtest engine for the Williams strategy.

The 5-minute :class:`~src.backtester.engine.BacktestEngine` enters at a bar's
close and never sees the overnight tape, so it cannot represent a daily system:
Williams enters on a resting stop order during the session and holds for days.
This engine models that directly.

What it models, bar by bar, in this order for every trading day:

1. **Overnight gap against an open position.** Stops are day orders working the
   regular session only, so an adverse overnight move is not filled at the stop
   price - it is filled at the next session's open. If the open is beyond the
   stop, the fill is the open and the trade is tagged ``gap_stop``. A gap
   through the target is filled at the open too (``gap_target``).
2. **Intraday path.** When 1-minute session paths are supplied, the entry
   minute and the exit minute are found on the actual path, so stop-versus-
   target ordering is resolved rather than assumed. Within a single minute bar
   the stop is assumed to fill first. Without 1-minute paths the fallback uses
   the session's high/low timestamps, and any remaining ambiguity resolves to
   the stop.
3. **Entry.** A buy stop fills at the trigger, or at the open when the session
   opens above the trigger.
4. **Time exit.** ``max_hold_days`` counts the entry day as day 1, so
   ``max_hold_days=1`` closes at the entry day's close and carries no overnight
   risk at all.
5. **Trailing exit on daily closes** (optional): after each close the stop is
   ratcheted to ``close - trailing_stop_points`` for a long.

Nothing is post-processed. Every P&L in the output is produced here.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.costs import CostModel
from src.backtester.engine import BacktestResult, Trade


@dataclass
class DailyTrade(Trade):
    """A completed daily-bar trade with gap-fill detail."""

    intended_stop: float = 0.0
    intended_target: float = 0.0
    gap_fill: bool = False
    gap_slippage_points: float = 0.0
    hold_days: int = 1
    entry_component: str = ""
    entry_gap_points: float = 0.0


@dataclass
class DailySimResult:
    """Output of one daily simulation over a contiguous span of days."""

    trades: list[DailyTrade] = field(default_factory=list)
    daily_pnl_points: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_net: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_gross: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    trading_days: int = 0

    @property
    def total_pnl_points(self) -> float:
        """Net P&L in points over the span."""
        return float(sum(t.pnl_net for t in self.trades))

    def to_backtest_result(self, strategy_name: str = "WilliamsStrategy") -> BacktestResult:
        """Wrap the result so the existing metrics and report code can read it.

        Args:
            strategy_name: Name recorded on the result.

        Returns:
            BacktestResult whose equity curve is the daily mark-to-market curve.
        """
        return BacktestResult(
            trades=list(self.trades),
            equity_gross=self.equity_gross,
            equity_net=self.equity_net,
            signals=self.signals,
            strategy_name=strategy_name,
        )


class IntradayPaths:
    """Per-session 1-minute high/low arrays used to order fills within a day."""

    def __init__(self, minute_df: pd.DataFrame, timezone: str = "US/Eastern",
                 session_start: str = "09:30", session_end: str = "16:00"):
        """Build the per-day path store from 1-minute bars.

        Args:
            minute_df: 1-minute OHLCV frame, tz-aware index, any capitalization.
            timezone: Session timezone.
            session_start: Session start HH:MM.
            session_end: Session end HH:MM (exclusive).
        """
        df = minute_df.rename(columns={c: c.lower() for c in minute_df.columns})
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        df = df.tz_convert(timezone).sort_index()

        def to_minutes(hhmm: str) -> int:
            hours, minutes = hhmm.split(":")
            return int(hours) * 60 + int(minutes)

        bar_minutes = df.index.hour * 60 + df.index.minute
        in_session = (bar_minutes >= to_minutes(session_start)) & (
            bar_minutes < to_minutes(session_end)
        )
        rth = df.loc[in_session]

        self._paths: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for day, group in rth.groupby(rth.index.normalize(), sort=False):
            self._paths[day.date()] = (
                group["high"].to_numpy(dtype=float),
                group["low"].to_numpy(dtype=float),
            )

    def get(self, timestamp) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (high, low) minute arrays for a session, or None if absent."""
        return self._paths.get(timestamp.date())


def _first_true(mask: np.ndarray, start: int) -> int:
    """Index of the first True at or after ``start``, or -1 if there is none."""
    if start >= len(mask):
        return -1
    tail = mask[start:]
    if not tail.any():
        return -1
    return int(np.argmax(tail)) + start


class DailyBacktestEngine:
    """Bar-by-bar simulator for daily stop-entry, multi-day-hold systems.

    Reads exit parameters from the strategy's ``params``:
    ``stop_points``, ``target_points``, ``max_hold_days`` and the optional
    ``trailing_stop_points``.
    """

    def __init__(
        self,
        strategy,
        cost_model: CostModel,
        intraday: IntradayPaths | None = None,
        close_at_end: bool = True,
    ):
        """Initialize the engine.

        Args:
            strategy: A WilliamsStrategy-like object exposing ``params`` and
                ``generate_signals``.
            cost_model: Slippage and commission model.
            intraday: Optional 1-minute session paths. When supplied, entry and
                exit minutes are located on the real path; when omitted the
                session high/low timestamps are used and ties go to the stop.
            close_at_end: Close any position still open on the final day of the
                simulated span at that day's close, so no trade straddles a
                split boundary.
        """
        self.strategy = strategy
        self.cost_model = cost_model
        self.intraday = intraday
        self.close_at_end = close_at_end

    # ---------------------------------------------------------------- helpers

    def _exit_after_entry(
        self,
        day: pd.Series,
        direction: int,
        stop: float,
        target: float,
        from_minute: int,
    ) -> tuple[str, float] | None:
        """Find the first stop or target hit at or after ``from_minute``.

        Args:
            day: The daily bar (needs high, low, high_time, low_time).
            direction: 1 long, -1 short.
            stop: Stop price.
            target: Target price.
            from_minute: Minute index the position is live from (0 for a day
                the position was already open at the session open).

        Returns:
            (exit_reason, exit_price) or None if neither level was reached.
        """
        paths = self.intraday.get(day.name) if self.intraday is not None else None

        if paths is not None:
            highs, lows = paths
            if direction == 1:
                stop_idx = _first_true(lows <= stop, from_minute)
                target_idx = _first_true(highs >= target, from_minute)
            else:
                stop_idx = _first_true(highs >= stop, from_minute)
                target_idx = _first_true(lows <= target, from_minute)

            if stop_idx < 0 and target_idx < 0:
                return None
            if target_idx < 0:
                return "stop_loss", stop
            if stop_idx < 0:
                return "take_profit", target
            # Same minute means the order inside that minute is unknown: assume
            # the stop filled first.
            if stop_idx <= target_idx:
                return "stop_loss", stop
            return "take_profit", target

        # Fallback: daily high/low plus the timestamps of those extremes.
        if direction == 1:
            hit_stop = day["low"] <= stop
            hit_target = day["high"] >= target
            target_first = day["high_time"] < day["low_time"]
        else:
            hit_stop = day["high"] >= stop
            hit_target = day["low"] <= target
            target_first = day["low_time"] < day["high_time"]

        if not hit_stop and not hit_target:
            return None
        if hit_stop and not hit_target:
            return "stop_loss", stop
        if hit_target and not hit_stop:
            return "take_profit", target
        return ("take_profit", target) if target_first else ("stop_loss", stop)

    def _entry_minute(self, day: pd.Series, direction: int, trigger: float) -> int:
        """Minute index at which a resting stop order at ``trigger`` filled.

        Args:
            day: Daily bar.
            direction: 1 long, -1 short.
            trigger: Stop-order price.

        Returns:
            Minute index of the fill, or 0 when the session opened beyond the
            trigger or when no 1-minute path is available.
        """
        if direction == 1 and day["open"] >= trigger:
            return 0
        if direction == -1 and day["open"] <= trigger:
            return 0

        paths = self.intraday.get(day.name) if self.intraday is not None else None
        if paths is None:
            return 0

        highs, lows = paths
        mask = highs >= trigger if direction == 1 else lows <= trigger
        idx = _first_true(mask, 0)
        return max(idx, 0)

    def _choose_entry(
        self, day: pd.Series, long_trigger: float, short_trigger: float
    ) -> tuple[int, float, int] | None:
        """Decide which resting stop order filled first on this session.

        Args:
            day: Daily bar.
            long_trigger: Buy-stop price, NaN if not armed.
            short_trigger: Sell-stop price, NaN if not armed.

        Returns:
            (direction, fill_price, fill_minute) or None if neither filled.
        """
        long_armed = not np.isnan(long_trigger)
        short_armed = not np.isnan(short_trigger)

        long_hit = long_armed and day["high"] >= long_trigger
        short_hit = short_armed and day["low"] <= short_trigger

        if not long_hit and not short_hit:
            return None

        if long_hit and short_hit:
            long_minute = self._entry_minute(day, 1, long_trigger)
            short_minute = self._entry_minute(day, -1, short_trigger)
            if self.intraday is None or self.intraday.get(day.name) is None:
                # No path available: use the session extreme timestamps.
                long_minute = 0 if day["high_time"] <= day["low_time"] else 1
                short_minute = 1 - long_minute
            if short_minute < long_minute:
                long_hit = False
            else:
                short_hit = False

        if long_hit:
            fill = max(float(day["open"]), float(long_trigger))
            return 1, fill, self._entry_minute(day, 1, long_trigger)

        fill = min(float(day["open"]), float(short_trigger))
        return -1, fill, self._entry_minute(day, -1, short_trigger)

    # ------------------------------------------------------------------- main

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        start: int = 0,
        end: int | None = None,
    ) -> DailySimResult:
        """Simulate the span ``df.iloc[start:end]`` day by day.

        Signals are generated over the full history so that indicator warm-up
        is not thrown away, but only the days inside the span can open or hold
        a position.

        Args:
            df: Full daily bar frame.
            signals: Frame with long_trigger / short_trigger columns aligned to df.
            start: First positional index of the span.
            end: One past the last positional index (defaults to len(df)).

        Returns:
            DailySimResult for the span.
        """
        if end is None:
            end = len(df)

        params = self.strategy.params
        stop_points = float(params["stop_points"])
        target_points = float(params["target_points"])
        max_hold_days = int(params["max_hold_days"])
        trailing_points = params.get("trailing_stop_points") or 0.0
        component_labels = signals.get("component")

        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["prior_close"]).abs(),
                (df["low"] - df["prior_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        avg_range = true_range.rolling(20, min_periods=1).mean()

        trades: list[DailyTrade] = []
        daily_pnl: dict[Any, float] = {}

        position = 0
        entry_price = 0.0
        entry_idx = 0
        entry_component = ""
        entry_gap = 0.0
        stop = 0.0
        target = 0.0
        days_held = 0

        for i in range(start, end):
            day = df.iloc[i]
            timestamp = df.index[i]
            day_points = 0.0
            last_close = df["close"].iloc[i - 1] if i > 0 else day["open"]

            # ---------------------------------------------------- open position
            if position != 0:
                days_held += 1
                exit_price = None
                reason = ""
                gap_fill = False

                # 1. Overnight gap. Stops are day orders, so an adverse
                #    overnight move is realized at this session's open.
                if position == 1 and day["open"] <= stop:
                    gap_fill = bool(day["open"] < stop)
                    exit_price = float(day["open"])
                    reason = "gap_stop" if gap_fill else "stop_loss"
                elif position == -1 and day["open"] >= stop:
                    gap_fill = bool(day["open"] > stop)
                    exit_price = float(day["open"])
                    reason = "gap_stop" if gap_fill else "stop_loss"
                elif position == 1 and day["open"] >= target:
                    exit_price, reason = float(day["open"]), "gap_target"
                elif position == -1 and day["open"] <= target:
                    exit_price, reason = float(day["open"]), "gap_target"

                # 2. Intraday path.
                if exit_price is None:
                    hit = self._exit_after_entry(day, position, stop, target, 0)
                    if hit is not None:
                        reason, exit_price = hit

                # 3. Time exit at the close.
                if exit_price is None and days_held >= max_hold_days:
                    exit_price, reason = float(day["close"]), "time_exit"

                # 4. Forced close on the final day of the span.
                if exit_price is None and self.close_at_end and i == end - 1:
                    exit_price, reason = float(day["close"]), "span_end"

                if exit_price is not None:
                    cost = self.cost_model.total_cost_per_trade(
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=position,
                        entry_volatility=float(true_range.iloc[entry_idx]),
                        exit_volatility=float(true_range.iloc[i]),
                        avg_volatility=float(avg_range.iloc[i]),
                    )
                    gross = (exit_price - entry_price) * position
                    intended = stop
                    slip = max(0.0, (intended - exit_price) * position) if gap_fill else 0.0

                    trades.append(
                        DailyTrade(
                            entry_idx=entry_idx,
                            exit_idx=i,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=position,
                            pnl_gross=gross,
                            pnl_net=gross - cost,
                            cost=cost,
                            entry_time=df.index[entry_idx],
                            exit_time=timestamp,
                            exit_reason=reason,
                            intended_stop=intended,
                            intended_target=target,
                            gap_fill=gap_fill,
                            gap_slippage_points=slip,
                            hold_days=days_held,
                            entry_component=entry_component,
                            entry_gap_points=entry_gap,
                        )
                    )
                    day_points += (exit_price - last_close) * position - cost
                    position = 0
                else:
                    day_points += (day["close"] - last_close) * position
                    if trailing_points:
                        if position == 1:
                            stop = max(stop, float(day["close"]) - trailing_points)
                        else:
                            stop = min(stop, float(day["close"]) + trailing_points)

            # --------------------------------------------------------- entries
            if position == 0:
                entry = self._choose_entry(
                    day,
                    float(signals["long_trigger"].iloc[i]),
                    float(signals["short_trigger"].iloc[i]),
                )
                if entry is not None:
                    direction, fill, minute = entry
                    position = direction
                    entry_price = fill
                    entry_idx = i
                    days_held = 1
                    entry_component = (
                        str(component_labels.iloc[i]) if component_labels is not None else ""
                    )
                    entry_gap = float(day["open"] - day["prior_close"]) if not pd.isna(
                        day["prior_close"]
                    ) else 0.0
                    stop = fill - stop_points * direction
                    target = fill + target_points * direction

                    exit_price = None
                    reason = ""
                    hit = self._exit_after_entry(day, direction, stop, target, minute)
                    if hit is not None:
                        reason, exit_price = hit
                    if exit_price is None and max_hold_days <= 1:
                        exit_price, reason = float(day["close"]), "time_exit"
                    if exit_price is None and self.close_at_end and i == end - 1:
                        exit_price, reason = float(day["close"]), "span_end"

                    if exit_price is not None:
                        cost = self.cost_model.total_cost_per_trade(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=direction,
                            entry_volatility=float(true_range.iloc[i]),
                            exit_volatility=float(true_range.iloc[i]),
                            avg_volatility=float(avg_range.iloc[i]),
                        )
                        gross = (exit_price - entry_price) * direction
                        trades.append(
                            DailyTrade(
                                entry_idx=i,
                                exit_idx=i,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                direction=direction,
                                pnl_gross=gross,
                                pnl_net=gross - cost,
                                cost=cost,
                                entry_time=timestamp,
                                exit_time=timestamp,
                                exit_reason=reason,
                                intended_stop=stop,
                                intended_target=target,
                                gap_fill=False,
                                gap_slippage_points=0.0,
                                hold_days=1,
                                entry_component=entry_component,
                                entry_gap_points=entry_gap,
                            )
                        )
                        day_points += gross - cost
                        position = 0
                    else:
                        day_points += (day["close"] - entry_price) * direction
                        if trailing_points:
                            if direction == 1:
                                stop = max(stop, float(day["close"]) - trailing_points)
                            else:
                                stop = min(stop, float(day["close"]) + trailing_points)

            daily_pnl[timestamp] = day_points

        pnl_series = pd.Series(daily_pnl, dtype=float)
        equity = pnl_series.cumsum()
        gross_series = pnl_series + pd.Series(
            {t.exit_time: t.cost for t in trades}, dtype=float
        ).reindex(pnl_series.index).fillna(0.0)

        return DailySimResult(
            trades=trades,
            daily_pnl_points=pnl_series,
            equity_net=equity,
            equity_gross=gross_series.cumsum(),
            signals=signals.iloc[start:end],
            trading_days=end - start,
        )


def gap_fill_statistics(trades: list[DailyTrade]) -> dict[str, Any]:
    """Summarize how often stops were gapped through and how much it cost.

    Args:
        trades: Trades from one or more daily simulations.

    Returns:
        Dict with the counts and the average / worst gap slippage in points.
    """
    losers = [t for t in trades if t.pnl_net < 0]
    gap_losers = [t for t in losers if t.gap_fill]
    slippages = [t.gap_slippage_points for t in gap_losers]
    overnight = [t for t in trades if t.hold_days > 1]

    return {
        "total_trades": len(trades),
        "overnight_trades": len(overnight),
        "losing_trades": len(losers),
        "gap_fill_losers": len(gap_losers),
        "gap_fill_share_of_losers": len(gap_losers) / len(losers) if losers else 0.0,
        "avg_gap_slippage_points": float(np.mean(slippages)) if slippages else 0.0,
        "worst_gap_slippage_points": float(np.max(slippages)) if slippages else 0.0,
        "worst_trade_points": min((t.pnl_net for t in trades), default=0.0),
    }


def daily_pnl_units(daily_pnl_points: pd.Series) -> list[Trade]:
    """Wrap each day's mark-to-market move as a resamplable unit.

    FundedNext's daily loss limit is a rule about a *day's equity move*, not
    about a trade. A multi-day trade that is gapped through its stop breaches
    the limit on the gap day, before the trader can act. Feeding one unit per
    trading day into the prop firm Monte Carlo makes the in-loop daily check
    test exactly that quantity.

    Args:
        daily_pnl_points: Per-day mark-to-market change in points, including
            days with no exposure (0.0).

    Returns:
        List of Trade objects, one per trading day.
    """
    units: list[Trade] = []
    for timestamp, points in daily_pnl_points.items():
        units.append(
            Trade(
                entry_idx=0,
                exit_idx=0,
                entry_price=0.0,
                exit_price=0.0,
                direction=1,
                pnl_gross=float(points),
                pnl_net=float(points),
                cost=0.0,
                entry_time=timestamp,
                exit_time=timestamp,
                exit_reason="daily_mark_to_market",
            )
        )
    return units


def replay_prop_account(
    daily_pnl_points: pd.Series,
    dollars_per_point: float,
    account_size: float,
    daily_loss_limit: float,
    equity_floor: float,
    profit_target: float,
) -> dict[str, Any]:
    """Replay the realized daily P&L through the prop firm rules, day by day.

    The daily loss limit and the equity floor are checked inside the loop. On a
    breach the account is dead and keeps the equity it had at that instant.
    Nothing is capped or re-scored afterwards.

    Args:
        daily_pnl_points: Realized per-day mark-to-market move in points.
        dollars_per_point: Dollar value of one point at the tested size.
        account_size: Starting equity.
        daily_loss_limit: Maximum permitted single-day loss.
        equity_floor: Hard equity floor.
        profit_target: Challenge profit target.

    Returns:
        Dict describing the outcome of the historical replay.
    """
    equity = account_size
    peak = account_size
    reached_target = False
    breach_day = None
    breach_reason = ""
    worst_day = 0.0

    for timestamp, points in daily_pnl_points.items():
        move = float(points) * dollars_per_point
        equity += move
        peak = max(peak, equity)
        worst_day = min(worst_day, move)

        if equity >= account_size + profit_target:
            reached_target = True

        if move < -daily_loss_limit:
            breach_day, breach_reason = timestamp, "daily loss limit"
            break
        if equity < equity_floor:
            breach_day, breach_reason = timestamp, "equity floor"
            break

    return {
        "survived": breach_day is None,
        "breach_day": breach_day,
        "breach_reason": breach_reason,
        "final_equity": equity,
        "peak_equity": peak,
        "reached_target": reached_target,
        "worst_day_dollars": worst_day,
        "days_traded": len(daily_pnl_points),
    }
