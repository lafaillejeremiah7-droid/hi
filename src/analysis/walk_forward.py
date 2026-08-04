"""Walk-Forward Analysis module.

Implements rolling/anchored walk-forward optimization to validate strategy
robustness. For each window:
1. Optimize parameters on the train portion
2. Lock params and run on the next OOS test month
3. Record trades and metrics for that OOS month
4. Concatenate all OOS months to get a walk-forward equity curve

This provides multiple independent OOS samples to guard against overfitting.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.metrics import compute_all_metrics, compute_sharpe_ratio
from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine, BacktestResult, Trade
from src.strategies.base import BaseStrategy


@dataclass
class WalkForwardWindow:
    """Results for a single walk-forward window."""

    window_idx: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_bars: int
    test_bars: int
    best_params: dict[str, Any]
    train_trades: list[Trade]
    test_trades: list[Trade]
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    train_equity: pd.Series
    test_equity: pd.Series


@dataclass
class WalkForwardResults:
    """Container for walk-forward analysis results."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    combined_oos_trades: list[Trade] = field(default_factory=list)
    combined_oos_equity: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    consistency_ratio: float = 0.0
    degradation_metric: float = 0.0
    walk_forward_efficiency: float = 0.0
    per_window_returns: list[float] = field(default_factory=list)
    regime_warning: bool = False
    total_oos_windows: int = 0
    profitable_windows: int = 0


class WalkForwardAnalyzer:
    """Walk-forward optimization analyzer.

    Implements walk-forward analysis with configurable rolling or anchored
    windows. Each window trains on historical data, locks parameters, then
    tests on a forward-looking OOS period.

    This ensures that every OOS month uses parameters optimized ONLY on
    data that preceded it -- no look-ahead bias is possible.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        cost_model: CostModel,
        train_window_months: int = 6,
        test_window_months: int = 1,
        step_months: int = 1,
        anchored: bool = False,
        max_trades_per_day: int | None = None,
        max_hold_bars: int | None = None,
        trading_session_start: str | None = None,
        trading_session_end: str | None = None,
        session_timezone: str = "US/Eastern",
        point_value: float = 20.0,
        exit_management: dict | None = None,
        close_at_end: bool = False,
    ):
        """Initialize walk-forward analyzer.

        Args:
            strategy: Trading strategy instance.
            cost_model: Cost model for slippage and commissions.
            train_window_months: Number of months for training window.
            test_window_months: Number of months for OOS test window.
            step_months: Step size in months between windows.
            anchored: If True, use expanding window (anchor at start).
                      If False, use rolling window.
            max_trades_per_day: Max trades per day limit.
            max_hold_bars: Deprecated, ignored.
            trading_session_start: Session start time.
            trading_session_end: Session end time.
            session_timezone: Timezone for session filtering.
            point_value: Dollar value per point.
            exit_management: Dict with exit management parameters.
            close_at_end: Close any position still open on a window's last bar.
        """
        self.strategy = strategy
        self.cost_model = cost_model
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.anchored = anchored
        self.max_trades_per_day = max_trades_per_day
        self.max_hold_bars = max_hold_bars
        self.trading_session_start = trading_session_start
        self.trading_session_end = trading_session_end
        self.session_timezone = session_timezone
        self.point_value = point_value
        self.exit_management = exit_management or {}
        self.close_at_end = close_at_end

    def _generate_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test window pairs from the data.

        Windows are generated chronologically with no overlap between
        a window's train and test periods.

        Args:
            df: Full preprocessed DataFrame with DatetimeIndex.

        Returns:
            List of (train_df, test_df) tuples.
        """
        windows = []

        if not hasattr(df.index, "to_pydatetime"):
            # Non-datetime index: use bar-based windowing
            return self._generate_windows_bar_based(df)

        start_date = df.index[0]
        end_date = df.index[-1]

        # Walk forward through the data
        current_test_start = start_date + pd.DateOffset(
            months=self.train_window_months
        )

        while True:
            test_end = current_test_start + pd.DateOffset(
                months=self.test_window_months
            )

            # Check if test window extends beyond data
            if current_test_start >= end_date:
                break

            # Define train window
            if self.anchored:
                train_start = start_date
            else:
                train_start = current_test_start - pd.DateOffset(
                    months=self.train_window_months
                )

            # Slice data
            train_mask = (df.index >= train_start) & (df.index < current_test_start)
            test_mask = (df.index >= current_test_start) & (df.index < test_end)

            train_df = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()

            # Only add window if both have sufficient data
            if len(train_df) >= 20 and len(test_df) >= 5:
                windows.append((train_df, test_df))

            # Step forward
            current_test_start = current_test_start + pd.DateOffset(
                months=self.step_months
            )

        return windows

    def _generate_windows_bar_based(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate windows based on bar count for non-datetime indexes.

        Uses approximate bars-per-month calculation.

        Args:
            df: DataFrame with any index type.

        Returns:
            List of (train_df, test_df) tuples.
        """
        # Estimate bars per month (assume ~21 trading days, bars depend on freq)
        n = len(df)
        if n < 50:
            return []

        # Rough estimate: total bars / total months
        if hasattr(df.index, "to_pydatetime"):
            total_days = (df.index[-1] - df.index[0]).days
            total_months = max(1, total_days / 30)
            bars_per_month = int(n / total_months)
        else:
            # Assume daily data
            bars_per_month = 21

        train_bars = bars_per_month * self.train_window_months
        test_bars = bars_per_month * self.test_window_months
        step_bars = bars_per_month * self.step_months

        windows = []
        pos = train_bars

        while pos + test_bars <= n:
            if self.anchored:
                train_start = 0
            else:
                train_start = pos - train_bars

            train_df = df.iloc[train_start:pos].copy()
            test_df = df.iloc[pos : pos + test_bars].copy()

            if len(train_df) >= 20 and len(test_df) >= 5:
                windows.append((train_df, test_df))

            pos += step_bars

        return windows

    def run(self, df: pd.DataFrame) -> WalkForwardResults:
        """Run walk-forward analysis on the full dataset.

        For each window:
        1. Optimize params on train portion
        2. Lock params and evaluate on OOS test portion
        3. Record all metrics

        Args:
            df: Full preprocessed DataFrame.

        Returns:
            WalkForwardResults with all windows, combined OOS equity, and metrics.
        """
        windows_data = self._generate_windows(df)

        if not windows_data:
            return WalkForwardResults()

        results_windows: list[WalkForwardWindow] = []
        all_oos_trades: list[Trade] = []
        all_oos_equities: list[pd.Series] = []
        per_window_returns: list[float] = []
        train_sharpes: list[float] = []
        test_sharpes: list[float] = []
        train_total_returns: list[float] = []
        test_total_returns: list[float] = []

        original_params = self.strategy.params.copy()

        for i, (train_df, test_df) in enumerate(windows_data):
            # Reset strategy params to original before each window
            self.strategy.params = original_params.copy()

            # Create engine for this window
            engine = BacktestEngine(
                strategy=self.strategy,
                cost_model=self.cost_model,
                initial_capital=100000.0,
                position_size=1,
                max_trades_per_day=self.max_trades_per_day,
                max_hold_bars=self.max_hold_bars,
                trading_session_start=self.trading_session_start,
                trading_session_end=self.trading_session_end,
                session_timezone=self.session_timezone,
                exit_management=self.exit_management,
                close_at_end=self.close_at_end,
            )

            # Optimize on train (use smaller subsample for speed in WF)
            # Override the engine's max_opt_bars for walk-forward efficiency
            opt_bars = min(1500, len(train_df))
            opt_df = train_df.iloc[-opt_bars:].copy() if len(train_df) > opt_bars else train_df
            best_params = engine._optimize_params(opt_df)
            if best_params:
                self.strategy.params.update(best_params)

            # Run on train with best params (use subsample for speed)
            train_eval_df = train_df.iloc[-opt_bars:].copy() if len(train_df) > opt_bars else train_df
            train_signals = self.strategy.generate_signals(train_eval_df)
            train_trades, train_eq_gross, train_eq_net = engine._simulate(
                train_eval_df, train_signals["signal"]
            )

            # Run on test (OOS) with locked params
            test_signals = self.strategy.generate_signals(test_df)
            test_trades, test_eq_gross, test_eq_net = engine._simulate(
                test_df, test_signals["signal"]
            )

            # Compute metrics
            train_result = BacktestResult(
                trades=train_trades,
                equity_gross=train_eq_gross,
                equity_net=train_eq_net,
                signals=train_signals,
                strategy_name=self.strategy.name,
            )
            test_result = BacktestResult(
                trades=test_trades,
                equity_gross=test_eq_gross,
                equity_net=test_eq_net,
                signals=test_signals,
                strategy_name=self.strategy.name,
            )

            train_metrics = compute_all_metrics(train_result, self.point_value)
            test_metrics = compute_all_metrics(test_result, self.point_value)

            # Record window
            window_result = WalkForwardWindow(
                window_idx=i,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                train_bars=len(train_df),
                test_bars=len(test_df),
                best_params=best_params.copy() if best_params else {},
                train_trades=train_trades,
                test_trades=test_trades,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                train_equity=train_eq_net,
                test_equity=test_eq_net,
            )
            results_windows.append(window_result)

            # Accumulate OOS data
            all_oos_trades.extend(test_trades)
            all_oos_equities.append(test_eq_net)

            # Track per-window returns
            test_return = float(test_eq_net.iloc[-1]) if len(test_eq_net) > 0 else 0.0
            per_window_returns.append(test_return)

            train_sharpes.append(train_metrics.get("sharpe_ratio", 0.0))
            test_sharpes.append(test_metrics.get("sharpe_ratio", 0.0))
            train_total_returns.append(train_metrics.get("total_return", 0.0))
            test_total_returns.append(test_metrics.get("total_return", 0.0))

        # Restore original params
        self.strategy.params = original_params

        # Build combined OOS equity curve (stitched)
        combined_oos_equity = self._stitch_equity_curves(all_oos_equities)

        # Compute summary metrics
        total_windows = len(results_windows)
        profitable_windows = sum(1 for r in per_window_returns if r > 0)
        consistency_ratio = (
            profitable_windows / total_windows if total_windows > 0 else 0.0
        )

        # Degradation metric: avg(OOS Sharpe) / avg(Train Sharpe)
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0.0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0
        degradation_metric = (
            avg_test_sharpe / avg_train_sharpe
            if avg_train_sharpe != 0
            else 0.0
        )

        # Walk-forward efficiency: total OOS return / total train return
        total_train_return = sum(train_total_returns)
        total_test_return = sum(test_total_returns)
        wf_efficiency = (
            total_test_return / total_train_return
            if total_train_return != 0
            else 0.0
        )

        # Regime warning: check if later months are worse than earlier
        regime_warning = False
        if len(per_window_returns) >= 4:
            first_half = per_window_returns[: len(per_window_returns) // 2]
            second_half = per_window_returns[len(per_window_returns) // 2 :]
            if np.mean(second_half) < np.mean(first_half) * 0.5:
                regime_warning = True

        return WalkForwardResults(
            windows=results_windows,
            combined_oos_trades=all_oos_trades,
            combined_oos_equity=combined_oos_equity,
            consistency_ratio=consistency_ratio,
            degradation_metric=degradation_metric,
            walk_forward_efficiency=wf_efficiency,
            per_window_returns=per_window_returns,
            regime_warning=regime_warning,
            total_oos_windows=total_windows,
            profitable_windows=profitable_windows,
        )

    def _stitch_equity_curves(
        self, equities: list[pd.Series]
    ) -> pd.Series:
        """Stitch multiple OOS equity curves into a continuous curve.

        Each OOS window's equity starts from 0 (relative P&L). We
        accumulate them sequentially to form a continuous curve.

        Args:
            equities: List of per-window OOS equity series.

        Returns:
            Combined equity series.
        """
        if not equities:
            return pd.Series(dtype=float)

        all_values = []
        all_indices = []
        cumulative_offset = 0.0

        for eq in equities:
            if len(eq) == 0:
                continue
            # Add the cumulative offset to this window's equity
            adjusted = eq + cumulative_offset
            all_values.extend(adjusted.values)
            all_indices.extend(eq.index)
            # Next window starts where this one ended
            cumulative_offset = float(adjusted.iloc[-1])

        if not all_values:
            return pd.Series(dtype=float)

        return pd.Series(all_values, index=all_indices)

    @classmethod
    def from_config(
        cls,
        strategy: BaseStrategy,
        config: dict,
        max_trades_per_day: int | None = None,
        max_hold_bars: int | None = None,
        trading_session_start: str | None = None,
        trading_session_end: str | None = None,
        session_timezone: str = "US/Eastern",
        close_at_end: bool = False,
    ) -> "WalkForwardAnalyzer":
        """Create WalkForwardAnalyzer from configuration dict.

        Args:
            strategy: Trading strategy instance.
            config: Full config dict with 'walk_forward' and 'costs' sections.
            max_trades_per_day: Max trades per day.
            max_hold_bars: Deprecated, ignored.
            trading_session_start: Session start.
            trading_session_end: Session end.
            session_timezone: Timezone.
            close_at_end: Close any position still open on a window's last bar.

        Returns:
            Configured WalkForwardAnalyzer instance.
        """
        wf_config = config.get("walk_forward", {})
        costs_config = config.get("costs", {})
        exit_mgmt_config = config.get("exit_management", {})

        cost_model = CostModel.from_config(costs_config)
        point_value = costs_config.get("point_value", 20.0)

        return cls(
            strategy=strategy,
            cost_model=cost_model,
            train_window_months=wf_config.get("train_window_months", 6),
            test_window_months=wf_config.get("test_window_months", 1),
            step_months=wf_config.get("step_months", 1),
            anchored=wf_config.get("anchored", False),
            max_trades_per_day=max_trades_per_day,
            max_hold_bars=max_hold_bars,
            trading_session_start=trading_session_start,
            trading_session_end=trading_session_end,
            session_timezone=session_timezone,
            point_value=point_value,
            exit_management=exit_mgmt_config,
            close_at_end=close_at_end,
        )
