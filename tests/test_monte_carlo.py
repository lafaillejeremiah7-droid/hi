"""Tests for Monte Carlo simulation module."""

import numpy as np
import pytest

from src.analysis.monte_carlo import MonteCarloSimulator, MonteCarloResults
from src.backtester.engine import Trade


def _make_trades(n_winners: int, n_losers: int, win_size: float = 5.0, loss_size: float = -3.0) -> list[Trade]:
    """Create synthetic trades for testing.

    Args:
        n_winners: Number of winning trades.
        n_losers: Number of losing trades.
        win_size: P&L for each winning trade (in points).
        loss_size: P&L for each losing trade (in points, negative).

    Returns:
        List of Trade objects.
    """
    trades = []
    for i in range(n_winners):
        trades.append(Trade(
            entry_idx=i * 2,
            exit_idx=i * 2 + 1,
            entry_price=100.0,
            exit_price=100.0 + win_size,
            direction=1,
            pnl_gross=win_size + 0.5,
            pnl_net=win_size,
            cost=0.5,
        ))
    for i in range(n_losers):
        trades.append(Trade(
            entry_idx=(n_winners + i) * 2,
            exit_idx=(n_winners + i) * 2 + 1,
            entry_price=100.0,
            exit_price=100.0 + loss_size,
            direction=1,
            pnl_gross=loss_size + 0.5,
            pnl_net=loss_size,
            cost=0.5,
        ))
    return trades


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def test_produces_correct_number_of_paths(self):
        """Simulator should generate exactly n_simulations paths."""
        trades = _make_trades(30, 20)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=500,
            seed=42,
        )
        results = sim.run_simulation()

        assert results.paths.shape[0] == 500
        assert results.n_simulations == 500

    def test_path_length_equals_n_trades_plus_one(self):
        """Each path should have n_trades + 1 points (including starting capital)."""
        trades = _make_trades(20, 10)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=100,
            seed=42,
        )
        results = sim.run_simulation()

        # Default n_trades = len(trades) = 30
        assert results.paths.shape[1] == 31  # 30 trades + 1 starting point
        assert results.n_trades_per_sim == 30

    def test_custom_n_trades(self):
        """Simulator should respect custom n_trades parameter."""
        trades = _make_trades(10, 5)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=100,
            seed=42,
        )
        results = sim.run_simulation(n_trades=50)

        assert results.paths.shape[1] == 51  # 50 trades + 1 starting
        assert results.n_trades_per_sim == 50

    def test_probability_of_ruin_between_0_and_1(self):
        """Probability of ruin should always be in [0, 1]."""
        trades = _make_trades(30, 20)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=1000,
            ruin_threshold=0.5,
            seed=42,
        )
        results = sim.run_simulation()

        assert 0.0 <= results.probability_of_ruin <= 1.0

    def test_probability_of_ruin_zero_for_all_winners(self):
        """With only winning trades, ruin probability should be 0."""
        trades = _make_trades(50, 0, win_size=10.0)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=1000,
            initial_capital=100000.0,
            ruin_threshold=0.5,
            seed=42,
        )
        results = sim.run_simulation()

        assert results.probability_of_ruin == 0.0

    def test_high_ruin_for_all_losers(self):
        """With only large losing trades, ruin probability should be high."""
        # Each trade loses $200 (10 pts * $20/pt), 50 trades = $10,000 loss
        # Ruin threshold 0.5 of $100,000 = must drop to $50,000
        # 50 trades * $200 = $10,000 total loss, not enough for ruin
        # Use bigger losses to ensure ruin
        trades = _make_trades(0, 50, loss_size=-100.0)  # Each loses 100*20 = $2000
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=1000,
            initial_capital=100000.0,
            point_value=20.0,
            ruin_threshold=0.5,
            seed=42,
        )
        results = sim.run_simulation()

        # 50 trades * $2000 = $100,000 total loss, definitely ruins
        assert results.probability_of_ruin > 0.9

    def test_median_outcome_reasonable(self):
        """Median should be close to expected value given trade statistics."""
        # 60% win rate, +5 pts wins, -3 pts losses
        # Expected per trade: 0.6*5*20 + 0.4*(-3)*20 = 60 - 24 = $36
        trades = _make_trades(60, 40, win_size=5.0, loss_size=-3.0)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=5000,
            initial_capital=100000.0,
            point_value=20.0,
            seed=42,
        )
        results = sim.run_simulation()

        # Expected final: 100000 + 100 * 36 = $103,600 (approx)
        # Allow wide margin due to random sampling
        assert results.median_final_equity > 100000.0  # Should be profitable
        assert results.median_final_equity < 110000.0  # Not unreasonably high

    def test_confidence_intervals_contain_median(self):
        """95% confidence interval should contain the median."""
        trades = _make_trades(40, 30)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=2000,
            confidence_level=0.95,
            seed=42,
        )
        results = sim.run_simulation()

        assert results.confidence_interval_lower <= results.median_final_equity
        assert results.confidence_interval_upper >= results.median_final_equity

    def test_confidence_interval_ordering(self):
        """Lower bound should be less than upper bound."""
        trades = _make_trades(25, 25)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=1000,
            confidence_level=0.95,
            seed=42,
        )
        results = sim.run_simulation()

        assert results.confidence_interval_lower <= results.confidence_interval_upper

    def test_worst_5pct_drawdown_positive(self):
        """Worst 5% drawdown should be non-negative."""
        trades = _make_trades(30, 20)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=1000,
            seed=42,
        )
        results = sim.run_simulation()

        assert results.worst_5pct_drawdown >= 0.0

    def test_max_drawdowns_array_length(self):
        """Max drawdowns array should have one value per simulation."""
        trades = _make_trades(20, 10)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=500,
            seed=42,
        )
        results = sim.run_simulation()

        assert len(results.max_drawdowns) == 500

    def test_paths_start_at_initial_capital(self):
        """All paths should start at initial_capital."""
        trades = _make_trades(20, 10)
        sim = MonteCarloSimulator(
            trades=trades,
            n_simulations=100,
            initial_capital=50000.0,
            seed=42,
        )
        results = sim.run_simulation()

        np.testing.assert_array_equal(results.paths[:, 0], 50000.0)

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        trades = _make_trades(30, 20)

        sim1 = MonteCarloSimulator(trades=trades, n_simulations=100, seed=123)
        results1 = sim1.run_simulation()

        sim2 = MonteCarloSimulator(trades=trades, n_simulations=100, seed=123)
        results2 = sim2.run_simulation()

        np.testing.assert_array_equal(results1.paths, results2.paths)
        assert results1.median_final_equity == results2.median_final_equity

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different paths."""
        trades = _make_trades(30, 20)

        sim1 = MonteCarloSimulator(trades=trades, n_simulations=100, seed=42)
        results1 = sim1.run_simulation()

        sim2 = MonteCarloSimulator(trades=trades, n_simulations=100, seed=99)
        results2 = sim2.run_simulation()

        # Paths should differ (extremely unlikely to be identical)
        assert not np.array_equal(results1.paths, results2.paths)

    def test_extract_trade_statistics(self):
        """Trade statistics should reflect input trade distribution."""
        trades = _make_trades(60, 40, win_size=5.0, loss_size=-3.0)
        sim = MonteCarloSimulator(trades=trades, n_simulations=100, point_value=20.0)
        stats = sim.extract_trade_statistics()

        assert stats["n_trades"] == 100
        assert abs(stats["win_rate"] - 0.6) < 0.01
        assert stats["win_mean"] == pytest.approx(5.0 * 20.0, abs=0.1)
        assert stats["loss_mean"] == pytest.approx(-3.0 * 20.0, abs=0.1)

    def test_empty_trades(self):
        """Simulator should handle empty trade list gracefully."""
        sim = MonteCarloSimulator(
            trades=[],
            n_simulations=100,
            initial_capital=100000.0,
            seed=42,
        )
        results = sim.run_simulation()

        assert results.median_final_equity == 100000.0
        assert results.probability_of_ruin == 0.0
        assert results.n_trades_per_sim == 0

    def test_from_config_factory(self):
        """Factory method should create simulator from config dict."""
        trades = _make_trades(20, 10)
        config = {
            "monte_carlo": {
                "simulations": 500,
                "confidence_level": 0.90,
                "ruin_threshold": 0.30,
                "seed": 99,
            },
            "costs": {
                "point_value": 20.0,
            },
        }

        sim = MonteCarloSimulator.from_config(trades=trades, config=config)

        assert sim.n_simulations == 500
        assert sim.confidence_level == 0.90
        assert sim.ruin_threshold == 0.30
        assert sim.seed == 99
        assert sim.point_value == 20.0

    def test_get_paths_runs_simulation_if_needed(self):
        """get_paths() should run simulation automatically if not already run."""
        trades = _make_trades(20, 10)
        sim = MonteCarloSimulator(trades=trades, n_simulations=50, seed=42)

        paths = sim.get_paths()
        assert paths.shape[0] == 50
        assert paths.shape[1] == 31  # 30 trades + 1

    def test_compute_results_idempotent(self):
        """Calling compute_results multiple times should return same result."""
        trades = _make_trades(20, 10)
        sim = MonteCarloSimulator(trades=trades, n_simulations=100, seed=42)

        r1 = sim.compute_results()
        r2 = sim.compute_results()

        assert r1.median_final_equity == r2.median_final_equity
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_final_equities_array_length(self):
        """Final equities array should match n_simulations."""
        trades = _make_trades(15, 10)
        sim = MonteCarloSimulator(trades=trades, n_simulations=750, seed=42)
        results = sim.run_simulation()

        assert len(results.final_equities) == 750
