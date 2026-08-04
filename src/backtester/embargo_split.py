"""Chronological three-way split with embargo gaps.

Different from every split used in this repo before: no interleaving, no
round-robin over calendar blocks. Straight chronological order, with a gap of
trading days thrown away between segments.

    | train 50% | embargo | validation 20% | embargo | true OOS ~30% |

The embargo exists because adjacent daily bars are autocorrelated and because a
Williams trade can be held for several days. Without a gap, the last trade of
train overlaps the first days of validation and the two segments share
information. Days inside an embargo are never simulated and no trade is
recorded there.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Segment:
    """A contiguous span of trading days."""

    name: str
    start_pos: int
    end_pos: int

    @property
    def n_days(self) -> int:
        """Number of trading days in the segment."""
        return self.end_pos - self.start_pos


def build_embargo_split(
    df: pd.DataFrame,
    train_fraction: float = 0.50,
    validation_fraction: float = 0.20,
    embargo_days: int = 5,
) -> list[Segment]:
    """Build the five segments of the embargoed chronological split.

    Args:
        df: Daily bar frame, chronologically sorted.
        train_fraction: Fraction of trading days assigned to train.
        validation_fraction: Fraction assigned to validation.
        embargo_days: Trading days discarded between segments.

    Returns:
        List of five Segments in order: train, embargo_1, validation,
        embargo_2, oos.
    """
    n = len(df)
    train_end = int(n * train_fraction)
    embargo_1_end = train_end + embargo_days
    validation_end = embargo_1_end + int(n * validation_fraction)
    embargo_2_end = validation_end + embargo_days

    if embargo_2_end >= n:
        raise ValueError(
            f"Split leaves no OOS days: {n} trading days is too few for "
            f"{train_fraction:.0%}/{validation_fraction:.0%} with "
            f"{embargo_days}-day embargoes"
        )

    return [
        Segment("train", 0, train_end),
        Segment("embargo_1", train_end, embargo_1_end),
        Segment("validation", embargo_1_end, validation_end),
        Segment("embargo_2", validation_end, embargo_2_end),
        Segment("oos", embargo_2_end, n),
    ]


def segment_by_name(segments: list[Segment], name: str) -> Segment:
    """Look a segment up by name.

    Args:
        segments: Segments from :func:`build_embargo_split`.
        name: Segment name.

    Returns:
        The matching Segment.
    """
    for segment in segments:
        if segment.name == name:
            return segment
    raise KeyError(f"No segment named {name!r}")


def print_embargo_audit(df: pd.DataFrame, segments: list[Segment]) -> None:
    """Print exact date ranges and day counts for all five segments.

    Args:
        df: Daily bar frame the split was built from.
        segments: Segments from :func:`build_embargo_split`.
    """
    total = len(df)
    print(f"\n  Chronological split with {segments[1].n_days}-trading-day embargo gaps")
    print(f"  Daily RTH bars: {total} trading days, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    print(f"\n  {'Segment':<12} {'Days':>6} {'Share':>7} {'First day':>12} {'Last day':>12}")
    print(f"  {'-' * 53}")

    for segment in segments:
        first = df.index[segment.start_pos].date()
        last = df.index[segment.end_pos - 1].date()
        print(f"  {segment.name:<12} {segment.n_days:>6} "
              f"{segment.n_days / total:>6.1%} {str(first):>12} {str(last):>12}")

    covered = sum(s.n_days for s in segments)
    print(f"  {'-' * 53}")
    print(f"  {'total':<12} {covered:>6} {covered / total:>6.1%}")
    print(f"  Embargo days are never simulated: {segments[1].n_days + segments[3].n_days} "
          f"trading days are discarded outright.")
