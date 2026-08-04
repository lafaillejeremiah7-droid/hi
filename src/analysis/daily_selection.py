"""Parameter selection for the daily Williams strategy.

Same discipline as :mod:`src.analysis.parameter_selection`, applied to the
daily engine and the embargoed chronological split:

1. Score every grid combination on TRAIN only.
2. Promote the top N by train score to VALIDATION.
3. Lock the validation winner. OOS is touched once, afterwards.

The grid is capped hard. Williams' claim is that a system needs very few
parameters, so a large search would contradict the thing being replicated.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import pandas as pd

from src.backtester.daily_engine import DailyBacktestEngine, DailySimResult
from src.backtester.embargo_split import Segment

MAX_GRID_COMBINATIONS = 40

# Parameters that change the signal series. Everything else only changes exits,
# so the signals can be cached across those combinations.
SIGNAL_KEYS = (
    "components",
    "gsv_lookback",
    "gsv_multiplier",
    "gsv_inverted",
    "smash_lookback",
    "tdom_filter",
)


@dataclass
class DailyComboScore:
    """Score for one parameter combination on one segment."""

    params: dict[str, Any]
    n_trades: int
    total_pnl_points: float
    ev_per_trade_points: float
    win_rate: float
    profit_factor: float
    max_consecutive_losers: int
    result: DailySimResult | None = field(default=None, repr=False)

    @property
    def score(self) -> float:
        """Selection score: total net P&L in points over the segment."""
        return self.total_pnl_points


def _signal_cache_key(params: dict[str, Any]) -> tuple:
    """Cache key covering every parameter that can change a signal."""
    key: list[Any] = []
    for name in SIGNAL_KEYS:
        value = params.get(name)
        key.append(tuple(value) if isinstance(value, (list, tuple)) else value)
    table = params.get("tdom_bias") or {}
    key.append(tuple(sorted(table.items())))
    return tuple(key)


def evaluate_daily_params(
    engine: DailyBacktestEngine,
    df: pd.DataFrame,
    segment: Segment,
    params: dict[str, Any],
    signal_cache: dict | None = None,
) -> DailyComboScore:
    """Run one parameter combination over one segment.

    Args:
        engine: Daily engine whose strategy is reconfigured in place.
        df: Full daily frame (signals are built over all of it so indicator
            warm-up is not lost; only the segment's days are simulated).
        segment: Segment to simulate.
        params: Parameter values to apply.
        signal_cache: Optional cache of signal frames.

    Returns:
        DailyComboScore including the underlying DailySimResult.
    """
    engine.strategy.params.update(params)

    key = _signal_cache_key(engine.strategy.params)
    if signal_cache is not None and key in signal_cache:
        signals = signal_cache[key]
    else:
        signals = engine.strategy.generate_signals(df)
        if signal_cache is not None:
            signal_cache[key] = signals

    result = engine.run(df, signals, start=segment.start_pos, end=segment.end_pos)

    trades = result.trades
    n_trades = len(trades)
    total = sum(t.pnl_net for t in trades)
    wins = [t.pnl_net for t in trades if t.pnl_net > 0]
    losses = [t.pnl_net for t in trades if t.pnl_net < 0]
    gross_loss = abs(sum(losses))

    streak = 0
    max_streak = 0
    for trade in trades:
        if trade.pnl_net < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        elif trade.pnl_net > 0:
            streak = 0

    return DailyComboScore(
        params=dict(params),
        n_trades=n_trades,
        total_pnl_points=total,
        ev_per_trade_points=total / n_trades if n_trades else 0.0,
        win_rate=len(wins) / n_trades if n_trades else 0.0,
        profit_factor=(sum(wins) / gross_loss) if gross_loss else float("inf") if wins else 0.0,
        max_consecutive_losers=max_streak,
        result=result,
    )


def build_daily_combos(
    grid: dict[str, list], derive_target: bool = True, target_r_multiple: float = 2.0
) -> list[dict[str, Any]]:
    """Expand the grid, ordered so signal parameters change slowest.

    Args:
        grid: Mapping of parameter name -> candidate values.
        derive_target: If True, set ``target_points`` from ``stop_points`` using
            ``target_r_multiple`` instead of tuning the target. This keeps the
            reward:risk ratio a declared constant rather than a free parameter.
        target_r_multiple: Fixed reward:risk multiple.

    Returns:
        List of parameter dicts.

    Raises:
        ValueError: If the grid exceeds MAX_GRID_COMBINATIONS.
    """
    keys = [k for k in SIGNAL_KEYS if k in grid]
    keys += [k for k in grid if k not in keys]

    combos = [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]

    if len(combos) > MAX_GRID_COMBINATIONS:
        raise ValueError(
            f"Grid has {len(combos)} combinations, above the hard cap of "
            f"{MAX_GRID_COMBINATIONS}. Trim the grid."
        )

    if derive_target:
        for combo in combos:
            if "stop_points" in combo:
                combo["target_points"] = combo["stop_points"] * target_r_multiple

    return combos


def select_daily_parameters(
    engine: DailyBacktestEngine,
    df: pd.DataFrame,
    train: Segment,
    validation: Segment,
    grid: dict[str, list],
    min_trades: int = 20,
    top_n: int = 3,
    target_r_multiple: float = 2.0,
) -> tuple[dict[str, Any], list[DailyComboScore], list[DailyComboScore]]:
    """Score the grid on train, re-score the finalists on validation, lock one.

    Args:
        engine: Daily engine wrapping a WilliamsStrategy.
        df: Full daily frame.
        train: Train segment.
        validation: Validation segment.
        grid: Parameter grid (at most MAX_GRID_COMBINATIONS combinations).
        min_trades: Minimum train trades for a combination to be eligible.
        top_n: How many train survivors are evaluated on validation.
        target_r_multiple: Fixed reward:risk multiple used to derive the target.

    Returns:
        Tuple of (locked params, all train scores best-first, validation scores
        for the finalists best-first).
    """
    combos = build_daily_combos(grid, target_r_multiple=target_r_multiple)
    signal_cache: dict = {}

    train_scores: list[DailyComboScore] = []
    for combo in combos:
        score = evaluate_daily_params(engine, df, train, combo, signal_cache)
        score.result = None  # keep memory flat across the grid
        train_scores.append(score)

    train_sorted = sorted(train_scores, key=lambda s: s.score, reverse=True)

    eligible = [s for s in train_sorted if s.n_trades >= min_trades and s.score > 0]
    if not eligible:
        # Nothing worth promoting. Return the best train combo so the caller can
        # report the failure honestly rather than crash.
        return dict(train_sorted[0].params), train_sorted, []

    finalists = eligible[:top_n]
    validation_scores = [
        evaluate_daily_params(engine, df, validation, s.params, signal_cache)
        for s in finalists
    ]
    validation_sorted = sorted(validation_scores, key=lambda s: s.score, reverse=True)

    return dict(validation_sorted[0].params), train_sorted, validation_sorted
