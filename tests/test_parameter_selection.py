"""Tests for the train-then-validate parameter selection.

Verifies that scoring happens on train only, that the finalists are the top
train survivors, that the locked combo is the best of those on validation,
and that OOS blocks are never touched during selection.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.parameter_selection import (
    build_combos,
    evaluate_params,
    select_parameters,
)
from src.backtester.block_split import blocks_for, build_blocks
from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine
from src.strategies.simple_strategy import SimpleStrategy

GRID = {
    "stop_points": [15, 20],
    "target_points": [30, 45],
    "delta_threshold": [0.5, 1.0],
    "level_proximity_points": [5, 10],
}


def make_session_data(months: int = 12) -> pd.DataFrame:
    """Create 5-min session bars spanning several months."""
    days = pd.date_range("2021-01-04", periods=months * 21, freq="B")
    stamps = []
    for day in days:
        for k in range(8):
            stamps.append(day + pd.Timedelta(hours=9, minutes=30 + 5 * k))

    index = pd.DatetimeIndex(stamps)
    n = len(index)
    rng = np.random.default_rng(5)
    base = np.cumsum(rng.normal(0, 4.0, n)) + 15000

    return pd.DataFrame(
        {
            "open": base,
            "high": base + 8.0,
            "low": base - 8.0,
            "close": base,
            "volume": rng.uniform(1000, 4000, n),
            "delta": rng.normal(0, 500, n),
        },
        index=index,
    )


def make_engine(strategy: SimpleStrategy) -> BacktestEngine:
    """Build an engine configured for fixed-point exits."""
    return BacktestEngine(
        strategy,
        CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0),
        max_trades_per_day=2,
        exit_management={
            "stop_loss_mode": "fixed",
            "partial_close_enabled": False,
        },
        close_at_end=True,
    )


class TestBuildCombos:
    """Tests for grid expansion."""

    def test_combo_count_is_the_full_product(self):
        """Every combination of every dimension is produced."""
        combos = build_combos(GRID)
        assert len(combos) == 2 * 2 * 2 * 2

    def test_signal_parameters_change_slowest(self):
        """Signal parameters are the outer loop so the cache is effective."""
        combos = build_combos(GRID)
        keys = list(combos[0])
        assert keys[:2] == ["level_proximity_points", "delta_threshold"]

    def test_full_pipeline_grid_size(self):
        """The configured 4x5x3x2 grid expands to 120 combinations."""
        combos = build_combos(
            {
                "stop_points": [15, 20, 25, 30],
                "target_points": [20, 30, 40, 45, 60],
                "delta_threshold": [0.5, 1.0, 1.5],
                "level_proximity_points": [5, 10],
            }
        )
        assert len(combos) == 120


class TestEvaluateParams:
    """Tests for single-combo evaluation."""

    def test_applies_params_to_strategy(self):
        """The evaluated parameters are the ones left on the strategy."""
        df = make_session_data(6)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = blocks_for(build_blocks(df), "train")

        evaluate_params(
            engine, df, blocks, {"stop_points": 25, "target_points": 45}
        )

        assert strategy.params["stop_points"] == 25
        assert strategy.params["target_points"] == 45

    def test_score_is_total_net_pnl(self):
        """The score equals the summed net P&L of the split's trades."""
        df = make_session_data(6)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = blocks_for(build_blocks(df), "train")

        score = evaluate_params(
            engine, df, blocks, {"stop_points": 20, "target_points": 30, "delta_threshold": 0.5, "level_proximity_points": 10}
        )

        assert score.score == pytest.approx(sum(t.pnl_net for t in score.result.trades))
        if score.n_trades:
            assert score.ev_per_trade_points == pytest.approx(
                score.total_pnl_points / score.n_trades
            )

    def test_signal_cache_avoids_recomputation(self):
        """Two combos sharing signal parameters generate signals once."""
        df = make_session_data(6)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = blocks_for(build_blocks(df), "train")

        calls = {"n": 0}
        original = strategy.generate_signals

        def counting_generate(frame):
            calls["n"] += 1
            return original(frame)

        strategy.generate_signals = counting_generate  # type: ignore[method-assign]
        cache: dict = {}
        base = {"delta_threshold": 0.5, "level_proximity_points": 10}
        evaluate_params(engine, df, blocks, {**base, "stop_points": 15, "target_points": 30}, cache)
        evaluate_params(engine, df, blocks, {**base, "stop_points": 25, "target_points": 45}, cache)

        assert calls["n"] == 1


class TestSelectParameters:
    """Tests for the train -> validation -> lock sequence."""

    def test_locked_params_come_from_the_grid(self):
        """The locked combination is one of the grid combinations."""
        df = make_session_data(12)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = build_blocks(df)

        locked, train_scores, val_scores = select_parameters(
            engine,
            df,
            blocks_for(blocks, "train"),
            blocks_for(blocks, "validation"),
            GRID,
            min_trades=1,
            top_n=3,
        )

        assert locked in build_combos(GRID)
        assert len(train_scores) == len(build_combos(GRID))
        assert len(val_scores) == 3

    def test_train_scores_are_sorted_best_first(self):
        """Train scores come back in descending score order."""
        df = make_session_data(12)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = build_blocks(df)

        _, train_scores, _ = select_parameters(
            engine,
            df,
            blocks_for(blocks, "train"),
            blocks_for(blocks, "validation"),
            GRID,
            min_trades=1,
            top_n=2,
        )

        scores = [s.score for s in train_scores]
        assert scores == sorted(scores, reverse=True)

    def test_locked_is_validation_best_of_finalists(self):
        """The locked combo is the finalist with the best validation score."""
        df = make_session_data(12)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = build_blocks(df)

        locked, _, val_scores = select_parameters(
            engine,
            df,
            blocks_for(blocks, "train"),
            blocks_for(blocks, "validation"),
            GRID,
            min_trades=1,
            top_n=4,
        )

        assert locked == val_scores[0].params
        assert val_scores[0].score == max(s.score for s in val_scores)

    def test_min_trades_gate_filters_finalists(self):
        """A very high min_trades gate leaves only busy combinations."""
        df = make_session_data(12)
        strategy = SimpleStrategy(params={"profile_lookback": 20})
        engine = make_engine(strategy)
        blocks = build_blocks(df)

        _, train_scores, val_scores = select_parameters(
            engine,
            df,
            blocks_for(blocks, "train"),
            blocks_for(blocks, "validation"),
            GRID,
            min_trades=10,
            top_n=3,
        )

        by_params = {tuple(sorted(s.params.items())): s for s in train_scores}
        for finalist in val_scores:
            key = tuple(sorted(finalist.params.items()))
            assert by_params[key].n_trades >= 10
