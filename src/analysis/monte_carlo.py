"""Monte Carlo simulation module.

Generates randomized trade sequences from empirical trade statistics
to assess strategy robustness, probability of ruin, and confidence intervals.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine import Trade


@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results."""

    n_simulations: int
    n_trades_per_sim: int
    median_final_equity: float
    mean_final_equity: float
    worst_5pct_drawdown: float
    probability_of_ruin: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    final_equities: np.ndarray = field(repr=False)
    paths: np.ndarray = field(repr=False)
    max_drawdowns: np.ndarray = field(repr=False)


class MonteCarloSimulator:
    """Monte Carlo simulation engine for trade sequence analysis.

    Takes completed trades from a backtest, extracts statistical properties
    (win rate, win/loss size distributions), and generates thousands of
    randomized trade sequences to assess strategy robustness.

    Attributes:
        trades: List of completed Trade objects.
        n_simulations: Number of simulation paths to generate.
        confidence_level: Confidence level for reporting (e.g. 0.95).
        ruin_threshold: Fraction of starting equity at which ruin occurs.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        trades: list[Trade],
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        ruin_threshold: float = 0.50,
        initial_capital: float = 100000.0,
        point_value: float = 20.0,
        seed: int | None = 42,
    ):
        """Initialize Monte Carlo simulator.

        Args:
            trades: List of completed Trade objects from backtest.
            n_simulations: Number of simulation paths.
            confidence_level: Confidence level for intervals (0-1).
            ruin_threshold: Fraction of peak equity that defines ruin.
            initial_capital: Starting capital in dollars.
            point_value: Dollar value per point.
            seed: Random seed (None for non-deterministic).
        """
        self.trades = trades
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.ruin_threshold = ruin_threshold
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.seed = seed

        # Extract trade P&L values in dollar terms
        self.trade_pnls = np.array([t.pnl_net * point_value for t in trades])

        # Results stored after run
        self._results: MonteCarloResults | None = None

    def extract_trade_statistics(self) -> dict[str, Any]:
        """Extract statistical properties from historical trades.

        Returns:
            Dictionary with win_rate, win_mean, win_std, loss_mean, loss_std,
            n_trades, avg_trade.
        """
        if len(self.trade_pnls) == 0:
            return {
                "win_rate": 0.0,
                "win_mean": 0.0,
                "win_std": 0.0,
                "loss_mean": 0.0,
                "loss_std": 0.0,
                "n_trades": 0,
                "avg_trade": 0.0,
            }

        winners = self.trade_pnls[self.trade_pnls > 0]
        losers = self.trade_pnls[self.trade_pnls <= 0]

        win_rate = len(winners) / len(self.trade_pnls) if len(self.trade_pnls) > 0 else 0.0

        return {
            "win_rate": win_rate,
            "win_mean": float(winners.mean()) if len(winners) > 0 else 0.0,
            "win_std": float(winners.std()) if len(winners) > 1 else 0.0,
            "loss_mean": float(losers.mean()) if len(losers) > 0 else 0.0,
            "loss_std": float(np.abs(losers).std()) if len(losers) > 1 else 0.0,
            "n_trades": len(self.trade_pnls),
            "avg_trade": float(self.trade_pnls.mean()),
        }

    def run_simulation(self, n_trades: int | None = None) -> MonteCarloResults:
        """Run Monte Carlo simulation.

        For each simulation path, randomly samples (with replacement) from
        the empirical trade P&L distribution and builds a cumulative
        equity curve starting from initial_capital.

        LIMITATION: This uses independent resampling (standard bootstrap),
        which discards serial correlation between consecutive trade outcomes.
        For a daily NAS100 strategy where trade results cluster by market
        regime (trending vs mean-reverting), independent resampling smooths
        out these clusters and produces narrower drawdown distributions than
        realized performance would show. The "worst 5% drawdown" is therefore
        an optimistic lower bound on actual tail risk.

        A block bootstrap (sampling contiguous blocks of k trades) would
        preserve local serial dependence while still randomizing the overall
        sequence. This is a well-known extension for future improvement.

        Args:
            n_trades: Number of trades per simulation path.
                     If None, uses the same number as the historical trades.

        Returns:
            MonteCarloResults with all computed statistics.
        """
        if len(self.trade_pnls) == 0:
            # No trades to simulate
            empty_paths = np.full((self.n_simulations, 1), self.initial_capital)
            self._results = MonteCarloResults(
                n_simulations=self.n_simulations,
                n_trades_per_sim=0,
                median_final_equity=self.initial_capital,
                mean_final_equity=self.initial_capital,
                worst_5pct_drawdown=0.0,
                probability_of_ruin=0.0,
                confidence_interval_lower=self.initial_capital,
                confidence_interval_upper=self.initial_capital,
                final_equities=np.full(self.n_simulations, self.initial_capital),
                paths=empty_paths,
                max_drawdowns=np.zeros(self.n_simulations),
            )
            return self._results

        if n_trades is None:
            n_trades = len(self.trade_pnls)

        rng = np.random.default_rng(self.seed)

        # Generate all random trade indices at once for efficiency
        # Shape: (n_simulations, n_trades)
        random_indices = rng.integers(0, len(self.trade_pnls), size=(self.n_simulations, n_trades))

        # Build equity paths
        # Each path starts at initial_capital and adds random trade P&Ls
        trade_returns = self.trade_pnls[random_indices]  # (n_sims, n_trades)

        # Cumulative sum of P&L + initial capital
        cumulative_pnl = np.cumsum(trade_returns, axis=1)
        paths = self.initial_capital + np.column_stack(
            [np.zeros(self.n_simulations), cumulative_pnl]
        )

        # Final equities
        final_equities = paths[:, -1]

        # Compute max drawdown for each path
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = running_max - paths
        max_drawdowns = drawdowns.max(axis=1)

        # Probability of ruin: fraction of paths where equity drops below threshold
        ruin_level = self.initial_capital * (1 - self.ruin_threshold)
        ruin_count = np.sum(paths.min(axis=1) <= ruin_level)
        probability_of_ruin = ruin_count / self.n_simulations

        # Confidence intervals on final equity
        alpha = 1 - self.confidence_level
        ci_lower = float(np.percentile(final_equities, alpha / 2 * 100))
        ci_upper = float(np.percentile(final_equities, (1 - alpha / 2) * 100))

        # Worst 5% drawdown
        worst_5pct_dd = float(np.percentile(max_drawdowns, 95))

        self._results = MonteCarloResults(
            n_simulations=self.n_simulations,
            n_trades_per_sim=n_trades,
            median_final_equity=float(np.median(final_equities)),
            mean_final_equity=float(np.mean(final_equities)),
            worst_5pct_drawdown=worst_5pct_dd,
            probability_of_ruin=float(probability_of_ruin),
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            final_equities=final_equities,
            paths=paths,
            max_drawdowns=max_drawdowns,
        )

        return self._results

    def compute_results(self) -> MonteCarloResults:
        """Compute and return results (runs simulation if not already run).

        Returns:
            MonteCarloResults object.
        """
        if self._results is None:
            return self.run_simulation()
        return self._results

    def get_paths(self) -> np.ndarray:
        """Get all simulated equity paths.

        Returns:
            Array of shape (n_simulations, n_trades + 1) with equity values.
        """
        if self._results is None:
            self.run_simulation()
        return self._results.paths

    @classmethod
    def from_config(
        cls,
        trades: list[Trade],
        config: dict,
        initial_capital: float = 100000.0,
    ) -> "MonteCarloSimulator":
        """Create MonteCarloSimulator from configuration dict.

        Args:
            trades: List of completed trades.
            config: Full config dict (with 'monte_carlo' and 'costs' sections).
            initial_capital: Starting capital.

        Returns:
            Configured MonteCarloSimulator instance.
        """
        mc_config = config.get("monte_carlo", {})
        costs_config = config.get("costs", {})

        return cls(
            trades=trades,
            n_simulations=mc_config.get("simulations", 10000),
            confidence_level=mc_config.get("confidence_level", 0.95),
            ruin_threshold=mc_config.get("ruin_threshold", 0.50),
            initial_capital=initial_capital,
            point_value=costs_config.get("point_value", 20.0),
            seed=mc_config.get("seed", 42),
        )
