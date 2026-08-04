"""Interleaved-block train / validation / OOS split.

The timeline is cut into consecutive calendar-month blocks which are then
assigned round-robin (train, train, validation, oos, repeat). Each split
therefore spans the whole 2021-2024 history instead of one contiguous era,
so no split is a single market regime.

Every block is simulated on its own, which guarantees a trade can never
straddle a block boundary: a position still open on a block's last bar is
closed at that bar's close (engine close_at_end) and recorded in that
block's split.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestEngine, BacktestResult, Trade

DEFAULT_ASSIGNMENT = ("train", "train", "validation", "oos")


@dataclass
class Block:
    """One contiguous block of bars assigned to a single split."""

    idx: int
    split: str
    start_pos: int  # inclusive positional index into the source DataFrame
    end_pos: int  # exclusive
    start_time: Any
    end_time: Any

    @property
    def n_bars(self) -> int:
        """Number of bars in the block."""
        return self.end_pos - self.start_pos


def build_blocks(
    df: pd.DataFrame,
    block_months: int = 1,
    assignment: tuple[str, ...] | list[str] = DEFAULT_ASSIGNMENT,
) -> list[Block]:
    """Cut the timeline into month blocks and assign them round-robin.

    Args:
        df: DataFrame with a DatetimeIndex, sorted ascending.
        block_months: Number of calendar months per block.
        assignment: Split names cycled over the blocks in order.

    Returns:
        List of Block records in chronological order.
    """
    if len(df) == 0:
        return []

    index = df.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    periods = index.to_period("M")
    unique_periods = list(pd.unique(periods))

    blocks: list[Block] = []
    for block_idx, offset in enumerate(range(0, len(unique_periods), block_months)):
        members = unique_periods[offset : offset + block_months]
        positions = np.flatnonzero(periods.isin(members))
        if len(positions) == 0:
            continue
        start_pos = int(positions[0])
        end_pos = int(positions[-1]) + 1
        blocks.append(
            Block(
                idx=block_idx,
                split=assignment[block_idx % len(assignment)],
                start_pos=start_pos,
                end_pos=end_pos,
                start_time=df.index[start_pos],
                end_time=df.index[end_pos - 1],
            )
        )

    return blocks


def blocks_for(blocks: list[Block], split: str) -> list[Block]:
    """Return the blocks assigned to a given split."""
    return [b for b in blocks if b.split == split]


def month_confounding_warning(
    blocks: list[Block], splits: tuple[str, ...] | list[str]
) -> str:
    """Describe calendar-month confounding introduced by the assignment cycle.

    A 4-long assignment cycle over 1-month blocks divides evenly into a
    12-month year, so every split receives the same calendar months in every
    year. That is worth stating out loud: month-of-year seasonality is then
    perfectly confounded with the split.

    Args:
        blocks: All blocks in chronological order.
        splits: Split names to inspect.

    Returns:
        Warning text, or an empty string when months are shared across splits.
    """
    months_by_split = {
        split: {b.start_time.month for b in blocks_for(blocks, split)}
        for split in splits
    }
    overlapping = any(
        months_by_split[a] & months_by_split[b]
        for i, a in enumerate(splits)
        for b in list(splits)[i + 1 :]
    )
    if overlapping:
        return ""

    lines = [
        "  WARNING: the assignment cycle divides evenly into the 12-month year,",
        "  so every split always gets the same calendar months:",
    ]
    for split in splits:
        names = ", ".join(
            pd.Timestamp(2021, m, 1).strftime("%b")
            for m in sorted(months_by_split[split])
        )
        lines.append(f"    {split:<11} {names}")
    lines.append(
        "  Month-of-year seasonality is therefore perfectly confounded with the"
    )
    lines.append(
        "  split. Train/validation/OOS differences may be seasonal, not real edge."
    )
    return "\n".join(lines)


def print_block_audit(blocks: list[Block], splits: tuple[str, ...] | list[str]) -> None:
    """Print the exact date range and bar count of every block per split.

    Args:
        blocks: All blocks in chronological order.
        splits: Split names to print, in the order they should appear.
    """
    total_bars = sum(b.n_bars for b in blocks)

    print(f"\n  Interleaved-block split: {len(blocks)} blocks, {total_bars:,} bars")
    print("  Assignment is round-robin over consecutive 1-month blocks.")

    warning = month_confounding_warning(blocks, splits)
    if warning:
        print()
        print(warning)

    for split in splits:
        split_blocks = blocks_for(blocks, split)
        split_bars = sum(b.n_bars for b in split_blocks)
        share = split_bars / total_bars if total_bars else 0.0
        print(
            f"\n  {split.upper()}: {len(split_blocks)} blocks, "
            f"{split_bars:,} bars ({share:.1%} of data)"
        )
        print(f"    {'Block':>5}  {'Start':<19} {'End':<19} {'Bars':>7}")
        for b in split_blocks:
            start = (
                b.start_time.strftime("%Y-%m-%d %H:%M")
                if hasattr(b.start_time, "strftime")
                else str(b.start_time)
            )
            end = (
                b.end_time.strftime("%Y-%m-%d %H:%M")
                if hasattr(b.end_time, "strftime")
                else str(b.end_time)
            )
            print(f"    {b.idx:>5}  {start:<19} {end:<19} {b.n_bars:>7,}")


def simulate_blocks(
    engine: BacktestEngine,
    df: pd.DataFrame,
    signals: pd.Series,
    blocks: list[Block],
) -> BacktestResult:
    """Simulate each block independently and concatenate the results.

    Signals are generated once on the full history (strictly causal) and then
    sliced per block, so no block needs its own indicator warmup. The
    simulation itself restarts flat at every block, so trades never straddle
    a boundary.

    Args:
        engine: Configured engine (close_at_end should be True).
        df: Full session-filtered DataFrame the signals were computed on.
        signals: Signal series aligned to df.
        blocks: Blocks belonging to a single split.

    Returns:
        BacktestResult with the split's trades and stitched equity curves.
    """
    all_trades: list[Trade] = []
    equity_gross_parts: list[pd.Series] = []
    equity_net_parts: list[pd.Series] = []
    offset_gross = 0.0
    offset_net = 0.0

    for block in blocks:
        block_df = df.iloc[block.start_pos : block.end_pos]
        block_signals = signals.iloc[block.start_pos : block.end_pos]

        trades, eq_gross, eq_net = engine._simulate(block_df, block_signals)

        all_trades.extend(trades)
        if len(eq_gross) > 0:
            equity_gross_parts.append(eq_gross + offset_gross)
            equity_net_parts.append(eq_net + offset_net)
            offset_gross = float(equity_gross_parts[-1].iloc[-1])
            offset_net = float(equity_net_parts[-1].iloc[-1])

    equity_gross = (
        pd.concat(equity_gross_parts) if equity_gross_parts else pd.Series(dtype=float)
    )
    equity_net = (
        pd.concat(equity_net_parts) if equity_net_parts else pd.Series(dtype=float)
    )

    return BacktestResult(
        trades=all_trades,
        equity_gross=equity_gross,
        equity_net=equity_net,
        signals=pd.DataFrame({"signal": signals}),
        strategy_name=engine.strategy.name,
    )
