"""Parameter selection for the SimpleStrategy.

The only tuning allowed in the pipeline:

1. Score every grid combination on TRAIN blocks only.
2. Take the top N by train score and evaluate exactly those on VALIDATION.
3. Lock the single best. OOS is touched once, afterwards, for reporting.

Signals only depend on two of the four grid dimensions (delta_threshold and
level_proximity_points), so signal series are cached and reused across the
stop / target combinations.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import pandas as pd

from src.backtester.block_split import Block, simulate_blocks
from src.backtester.engine import BacktestEngine, BacktestResult

# Signal parameters first so the cache is hit for every stop/target pair.
SIGNAL_KEYS = ("level_proximity_points", "delta_threshold")
EXIT_KEYS = ("stop_points", "target_points")


@dataclass
class ComboScore:
    """Score for one parameter combination on one split."""

    params: dict[str, Any]
    n_trades: int
    total_pnl_points: float
    ev_per_trade_points: float
    win_rate: float
    result: BacktestResult | None = field(default=None, repr=False)

    @property
    def score(self) -> float:
        """Selection score: total net P&L in points over the split."""
        return self.total_pnl_points


def evaluate_params(
    engine: BacktestEngine,
    df: pd.DataFrame,
    blocks: list[Block],
    params: dict[str, Any],
    signal_cache: dict | None = None,
) -> ComboScore:
    """Run one parameter combination over a set of blocks.

    Args:
        engine: Engine whose strategy will be reconfigured in place.
        df: Full session-filtered DataFrame.
        blocks: Blocks belonging to the split being evaluated.
        params: Parameter values to apply to the strategy.
        signal_cache: Optional cache keyed by the signal parameters.

    Returns:
        ComboScore including the underlying BacktestResult.
    """
    strategy = engine.strategy
    strategy.params.update(params)

    cache_key = tuple(strategy.params[k] for k in SIGNAL_KEYS) + (
        strategy.params["profile_lookback"],
    )
    if signal_cache is not None and cache_key in signal_cache:
        signals = signal_cache[cache_key]
    else:
        signals = strategy.generate_signals(df)["signal"]
        if signal_cache is not None:
            signal_cache[cache_key] = signals

    result = simulate_blocks(engine, df, signals, blocks)

    trades = result.trades
    n_trades = len(trades)
    total_pnl = sum(t.pnl_net for t in trades)
    wins = sum(1 for t in trades if t.pnl_net > 0)

    return ComboScore(
        params=dict(params),
        n_trades=n_trades,
        total_pnl_points=total_pnl,
        ev_per_trade_points=total_pnl / n_trades if n_trades else 0.0,
        win_rate=wins / n_trades if n_trades else 0.0,
        result=result,
    )


def build_combos(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand the grid, ordered so signal parameters change slowest.

    Args:
        grid: Mapping of parameter name -> list of values.

    Returns:
        List of parameter dicts.
    """
    keys = [k for k in SIGNAL_KEYS + EXIT_KEYS if k in grid]
    keys += [k for k in grid if k not in keys]
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def select_parameters(
    engine: BacktestEngine,
    df: pd.DataFrame,
    train_blocks: list[Block],
    validation_blocks: list[Block],
    grid: dict[str, list],
    min_trades: int = 30,
    top_n: int = 5,
) -> tuple[dict[str, Any], list[ComboScore], list[ComboScore]]:
    """Score the grid on train, re-score the survivors on validation, lock one.

    Args:
        engine: Engine wrapping a SimpleStrategy instance.
        df: Full session-filtered DataFrame.
        train_blocks: Blocks assigned to train.
        validation_blocks: Blocks assigned to validation.
        grid: Parameter grid.
        min_trades: Minimum train trades for a combo to be eligible.
        top_n: How many train survivors are evaluated on validation.

    Returns:
        Tuple of (locked params, all train scores sorted best-first,
        validation scores for the finalists sorted best-first).
    """
    combos = build_combos(grid)
    signal_cache: dict = {}

    train_scores: list[ComboScore] = []
    for combo in combos:
        score = evaluate_params(engine, df, train_blocks, combo, signal_cache)
        score.result = None  # keep memory flat across the grid
        train_scores.append(score)

    eligible = [s for s in train_scores if s.n_trades >= min_trades]
    if not eligible:
        eligible = train_scores

    train_scores_sorted = sorted(train_scores, key=lambda s: s.score, reverse=True)
    finalists = sorted(eligible, key=lambda s: s.score, reverse=True)[:top_n]

    validation_scores = [
        evaluate_params(engine, df, validation_blocks, s.params, signal_cache)
        for s in finalists
    ]
    validation_sorted = sorted(validation_scores, key=lambda s: s.score, reverse=True)

    locked = dict(validation_sorted[0].params)
    return locked, train_scores_sorted, validation_sorted
