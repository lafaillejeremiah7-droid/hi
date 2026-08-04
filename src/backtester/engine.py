"""Backtesting engine.

Processes strategy signals bar-by-bar, maintains position state,
applies costs, and implements train/validation split with parameter
optimization. Supports scalping features: session filtering,
max trades per day, and time-based exits.

Supports two stop loss modes:
- "atr": ATR-based stops
- "fixed": Fixed-point stops

Partial close and trailing are optional (partial_close_enabled). With them
disabled the simulation is a plain fixed stop / fixed target system.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.costs import CostModel
from src.strategies.base import BaseStrategy


@dataclass
class Trade:
    """Represents a completed trade."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: int  # 1=long, -1=short
    pnl_gross: float  # P&L without costs (in points)
    pnl_net: float  # P&L with costs (in points)
    cost: float  # Total costs (in points)
    entry_time: Any = None
    exit_time: Any = None
    exit_reason: str = ""  # stop_loss, take_profit, trailing_stop, partial_then_stop, block_end
    partial_close_pnl: float = 0.0  # P&L from the 50% partial close
    trailing_exit_pnl: float = 0.0  # P&L from the remaining 50% trailing exit


@dataclass
class BacktestResult:
    """Container for backtest results."""

    trades: list[Trade] = field(default_factory=list)
    equity_gross: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_net: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    signals: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    train_result: "BacktestResult | None" = None
    validation_result: "BacktestResult | None" = None
    oos_result: "BacktestResult | None" = None
    best_params: dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""


class BacktestEngine:
    """Event-driven backtesting engine.

    Processes signals bar-by-bar, tracking position state and
    applying entry/exit logic:
    - Stop loss and take profit come from the strategy
    - Partial close: 50% at configured profit distance (optional)
    - Trailing stop: activated after partial close (optional)

    Produces dual equity curves (with and without costs).

    Supports train/validation split with grid search parameter
    optimization on the training set.

    Features:
    - Session filter: only trade during specified hours
    - Max trades per day: cap daily entries
    - Optional partial close + trailing stop
    - Optional forced close of any open position on the final bar
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        cost_model: CostModel,
        initial_capital: float = 100000.0,
        position_size: int = 1,
        max_trades_per_day: int | None = None,
        max_hold_bars: int | None = None,
        trading_session_start: str | None = None,
        trading_session_end: str | None = None,
        session_timezone: str = "US/Eastern",
        exit_management: dict | None = None,
        close_at_end: bool = False,
    ):
        """Initialize backtest engine.

        Args:
            strategy: Trading strategy instance.
            cost_model: Cost model for slippage and commissions.
            initial_capital: Starting capital in dollars.
            position_size: Number of contracts per trade (fixed).
            max_trades_per_day: Maximum trades allowed per calendar day.
            max_hold_bars: Deprecated, ignored. Kept for API compatibility.
            trading_session_start: Session start time (HH:MM format).
            trading_session_end: Session end time (HH:MM format).
            session_timezone: Timezone for session filtering.
            exit_management: Dict with exit management parameters.
            close_at_end: If True, any position still open on the final bar is
                closed at that bar's close (used for block-based splits so a
                trade can never straddle a block boundary).
        """
        self.strategy = strategy
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_trades_per_day = max_trades_per_day
        # max_hold_bars is deprecated - no longer used for forced exits
        self.max_hold_bars = None
        self.trading_session_start = trading_session_start
        self.trading_session_end = trading_session_end
        self.session_timezone = session_timezone
        self.close_at_end = close_at_end

        # Exit management parameters
        em = exit_management or {}
        self.stop_loss_mode = em.get("stop_loss_mode", "atr")
        self.stop_loss_atr_mult = em.get("stop_loss_atr_mult", 1.0)
        self.take_profit_atr_mult = em.get("take_profit_atr_mult", 1.5)
        self.extended_take_profit_atr_mult = em.get("extended_take_profit_atr_mult", 2.5)
        self.feature_zscore_threshold = em.get("feature_zscore_threshold", 2.5)
        self.partial_close_atr_mult = em.get("partial_close_atr_mult", 1.0)
        self.partial_close_fraction = em.get("partial_close_fraction", 0.5)
        self.trailing_stop_atr_mult = em.get("trailing_stop_atr_mult", 1.0)
        # Fixed-point parameters
        self.stop_loss_fixed_points = em.get("stop_loss_fixed_points", 20)
        self.partial_close_fixed_points = em.get("partial_close_fixed_points", 20)
        self.trailing_stop_fixed_points = em.get("trailing_stop_fixed_points", 20)
        # When False the position runs to the fixed stop or fixed target only.
        self.partial_close_enabled = em.get("partial_close_enabled", True)

    def _is_within_session(self, timestamp) -> bool:
        """Check if a timestamp is within the trading session.

        Args:
            timestamp: The bar's timestamp.

        Returns:
            True if within session or no session filter configured.
        """
        if self.trading_session_start is None or self.trading_session_end is None:
            return True

        if not hasattr(timestamp, "hour"):
            return True

        # Parse session times
        start_parts = self.trading_session_start.split(":")
        end_parts = self.trading_session_end.split(":")
        start_hour, start_min = int(start_parts[0]), int(start_parts[1])
        end_hour, end_min = int(end_parts[0]), int(end_parts[1])

        bar_hour = timestamp.hour
        bar_min = timestamp.minute

        # Convert to minutes since midnight for comparison
        bar_minutes = bar_hour * 60 + bar_min
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min

        return start_minutes <= bar_minutes < end_minutes

    def _get_bar_date(self, timestamp) -> Any:
        """Extract date from a timestamp for daily trade counting."""
        if hasattr(timestamp, "date"):
            return timestamp.date()
        return None

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on the provided data.

        Args:
            df: Preprocessed DataFrame with OHLCV and indicator columns.

        Returns:
            BacktestResult with trades, equity curves, and signals.
        """
        # Generate signals
        signals_df = self.strategy.generate_signals(df)
        signals = signals_df["signal"]

        # Run simulation
        trades, equity_gross, equity_net = self._simulate(df, signals)

        return BacktestResult(
            trades=trades,
            equity_gross=equity_gross,
            equity_net=equity_net,
            signals=signals_df,
            strategy_name=self.strategy.name,
        )

    def run_with_split(
        self, df: pd.DataFrame, train_ratio: float = 0.6
    ) -> BacktestResult:
        """Run backtest with train/validation split and optimization.

        1. Split data into train and validation periods.
        2. Optimize parameters via grid search on train set.
        3. Run with best parameters on validation set.

        Args:
            df: Full preprocessed DataFrame.
            train_ratio: Fraction of data for training (0-1).

        Returns:
            BacktestResult with train_result, validation_result, and best_params.
        """
        # Split data
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        # Optimize on train set
        best_params = self._optimize_params(train_df)

        # Apply best parameters
        if best_params:
            self.strategy.params.update(best_params)

        # Run on train set with best params
        train_signals = self.strategy.generate_signals(train_df)
        train_trades, train_eq_gross, train_eq_net = self._simulate(
            train_df, train_signals["signal"]
        )
        train_result = BacktestResult(
            trades=train_trades,
            equity_gross=train_eq_gross,
            equity_net=train_eq_net,
            signals=train_signals,
            strategy_name=self.strategy.name,
        )

        # Run on validation set with same params
        val_signals = self.strategy.generate_signals(val_df)
        val_trades, val_eq_gross, val_eq_net = self._simulate(
            val_df, val_signals["signal"]
        )
        val_result = BacktestResult(
            trades=val_trades,
            equity_gross=val_eq_gross,
            equity_net=val_eq_net,
            signals=val_signals,
            strategy_name=self.strategy.name,
        )

        # Combined result
        full_signals = self.strategy.generate_signals(df)
        full_trades, full_eq_gross, full_eq_net = self._simulate(
            df, full_signals["signal"]
        )

        return BacktestResult(
            trades=full_trades,
            equity_gross=full_eq_gross,
            equity_net=full_eq_net,
            signals=full_signals,
            train_result=train_result,
            validation_result=val_result,
            best_params=best_params,
            strategy_name=self.strategy.name,
        )

    def run_with_three_way_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.50,
        validation_ratio: float = 0.25,
    ) -> BacktestResult:
        """Run backtest with train/validation/OOS three-way split.

        1. Split data into train (first), validation (middle), and OOS (final).
        2. Optimize parameters via grid search on train set.
        3. Select best params based on validation performance.
        4. Run final evaluation on OOS with locked params (never touched before).

        Args:
            df: Full preprocessed DataFrame.
            train_ratio: Fraction of data for training (0-1).
            validation_ratio: Fraction of data for validation (0-1).

        Returns:
            BacktestResult with train_result, validation_result, oos_result, and best_params.
        """
        # Split data three ways
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        oos_df = df.iloc[val_end:].copy()

        print(f"    [3-Way Split] Train: {len(train_df)} bars "
              f"({df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else df.index[0]} to "
              f"{df.index[train_end-1].strftime('%Y-%m-%d') if hasattr(df.index[train_end-1], 'strftime') else df.index[train_end-1]})")
        print(f"    [3-Way Split] Validation: {len(val_df)} bars "
              f"({df.index[train_end].strftime('%Y-%m-%d') if hasattr(df.index[train_end], 'strftime') else df.index[train_end]} to "
              f"{df.index[val_end-1].strftime('%Y-%m-%d') if hasattr(df.index[val_end-1], 'strftime') else df.index[val_end-1]})")
        print(f"    [3-Way Split] True OOS: {len(oos_df)} bars "
              f"({df.index[val_end].strftime('%Y-%m-%d') if hasattr(df.index[val_end], 'strftime') else df.index[val_end]} to "
              f"{df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else df.index[-1]})")

        # Step 1: Optimize on train set
        best_params = self._optimize_params(train_df)

        # Apply best parameters
        if best_params:
            self.strategy.params.update(best_params)

        # Step 2: Run on train set with best params
        train_signals = self.strategy.generate_signals(train_df)
        train_trades, train_eq_gross, train_eq_net = self._simulate(
            train_df, train_signals["signal"]
        )
        train_result = BacktestResult(
            trades=train_trades,
            equity_gross=train_eq_gross,
            equity_net=train_eq_net,
            signals=train_signals,
            strategy_name=self.strategy.name,
        )

        # Step 3: Run on validation set with locked params
        val_signals = self.strategy.generate_signals(val_df)
        val_trades, val_eq_gross, val_eq_net = self._simulate(
            val_df, val_signals["signal"]
        )
        val_result = BacktestResult(
            trades=val_trades,
            equity_gross=val_eq_gross,
            equity_net=val_eq_net,
            signals=val_signals,
            strategy_name=self.strategy.name,
        )

        # Step 4: Run on TRUE OOS set with locked params (NEVER seen before)
        oos_signals = self.strategy.generate_signals(oos_df)
        oos_trades, oos_eq_gross, oos_eq_net = self._simulate(
            oos_df, oos_signals["signal"]
        )
        oos_result = BacktestResult(
            trades=oos_trades,
            equity_gross=oos_eq_gross,
            equity_net=oos_eq_net,
            signals=oos_signals,
            strategy_name=self.strategy.name,
        )

        # Combined result over full dataset (reuse already-computed signals)
        # Instead of regenerating signals on the full dataset, combine the
        # already-computed partial results for efficiency
        full_trades = train_trades + val_trades + oos_trades
        full_eq_gross = pd.concat([train_eq_gross, val_eq_gross, oos_eq_gross])
        # Adjust equity curves to be cumulative across splits
        val_offset = float(train_eq_gross.iloc[-1]) if len(train_eq_gross) > 0 else 0.0
        oos_offset = val_offset + (float(val_eq_gross.iloc[-1]) if len(val_eq_gross) > 0 else 0.0)
        full_eq_gross = pd.concat([
            train_eq_gross,
            val_eq_gross + val_offset,
            oos_eq_gross + oos_offset,
        ])
        full_eq_net = pd.concat([
            train_eq_net,
            val_eq_net + val_offset,
            oos_eq_net + oos_offset,
        ])
        full_signals = pd.concat([train_signals, val_signals, oos_signals])

        return BacktestResult(
            trades=full_trades,
            equity_gross=full_eq_gross,
            equity_net=full_eq_net,
            signals=full_signals,
            train_result=train_result,
            validation_result=val_result,
            oos_result=oos_result,
            best_params=best_params,
            strategy_name=self.strategy.name,
        )

    def _simulate(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> tuple[list[Trade], pd.Series, pd.Series]:
        """Simulate trades bar-by-bar.

        Args:
            df: DataFrame with OHLCV data.
            signals: Series of signals (1, -1, 0).

        Returns:
            Tuple of (trades list, gross equity curve, net equity curve).
        """
        return self._simulate_classic(df, signals)

    def _simulate_classic(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> tuple[list[Trade], pd.Series, pd.Series]:
        """Simulate trades bar-by-bar.

        Implements:
        - Session filtering (skip signals outside trading hours)
        - Max trades per day (skip signals once daily limit reached)
        - Stop loss and take profit taken from the strategy
        - Partial close + trailing stop when partial_close_enabled is True
        - Forced close on the final bar when close_at_end is True
        - No max_hold forced exit

        Position phases (only when partial_close_enabled):
        - FULL: 100% position, waiting for partial close trigger or SL/TP
        - PARTIAL: 50% remaining after partial close, trailing stop active

        Args:
            df: DataFrame with OHLCV data.
            signals: Series of signals (1, -1, 0).

        Returns:
            Tuple of (trades list, gross equity curve, net equity curve).
        """
        trades: list[Trade] = []
        n = len(df)
        equity_gross_arr = np.zeros(n)
        equity_net_arr = np.zeros(n)

        # Compute volatility for cost model (vectorized)
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        close_shift = np.empty(n)
        close_shift[0] = close[0]
        close_shift[1:] = close[:-1]

        tr_arr = np.maximum(
            high - low,
            np.maximum(np.abs(high - close_shift), np.abs(low - close_shift))
        )

        # Rolling average volatility (simple cumulative approach for speed)
        avg_volatility = pd.Series(tr_arr).rolling(window=20, min_periods=1).mean().values

        # Pre-compute ATR for partial close / trailing stop calculations
        atr_period = 14
        atr_arr = pd.Series(tr_arr).rolling(window=atr_period, min_periods=1).mean().values

        # Pre-compute session mask (vectorized)
        if self.trading_session_start is not None and self.trading_session_end is not None and hasattr(df.index, "hour"):
            start_parts = self.trading_session_start.split(":")
            end_parts = self.trading_session_end.split(":")
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
            end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
            bar_minutes = df.index.hour * 60 + df.index.minute
            in_session_arr = np.asarray((bar_minutes >= start_minutes) & (bar_minutes < end_minutes))
        else:
            in_session_arr = np.ones(n, dtype=bool)

        # Pre-compute dates for daily trade counting
        if hasattr(df.index, "date"):
            dates_arr = df.index.date
        else:
            dates_arr = None

        # Pre-extract signal values as numpy array
        signals_arr = signals.values.astype(int)

        # Pre-extract delta_zscore if available
        if "delta_zscore" in df.columns:
            delta_zscore_arr = df["delta_zscore"].values
        else:
            delta_zscore_arr = np.zeros(n)

        # Position state
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_idx = 0
        stop_loss = 0.0
        take_profit = 0.0
        bars_in_trade = 0

        # Dynamic exit state
        partial_close_done = False
        partial_close_pnl = 0.0
        partial_close_cost = 0.0
        best_price_since_entry = 0.0
        trailing_stop_price = 0.0
        partial_close_trigger = 0.0  # Price where partial close triggers
        entry_atr = 0.0
        entry_zscore = 0.0

        # Daily trade tracking
        daily_trade_count: dict = {}

        cumulative_gross = 0.0
        cumulative_net = 0.0

        for i in range(n):
            current_signal = signals_arr[i]
            in_session = in_session_arr[i]
            current_date = dates_arr[i] if dates_arr is not None else None

            # Check daily trade limit
            trades_today = daily_trade_count.get(current_date, 0) if current_date else 0
            daily_limit_reached = (
                self.max_trades_per_day is not None
                and trades_today >= self.max_trades_per_day
            )

            if position == 0:
                # No position - check for entry signal
                if current_signal in (1, -1) and in_session and not daily_limit_reached:
                    position = int(current_signal)
                    entry_price = close[i]
                    entry_idx = i
                    bars_in_trade = 0
                    entry_atr = atr_arr[i]
                    entry_zscore = delta_zscore_arr[i]

                    # Compute stop loss
                    stop_loss = self.strategy.get_stop_loss(df, i, position)

                    # Compute take profit with feature Z-score
                    take_profit = self.strategy.get_take_profit(
                        df, i, position, feature_zscore=entry_zscore
                    )

                    # Initialize dynamic exit state
                    partial_close_done = False
                    partial_close_pnl = 0.0
                    partial_close_cost = 0.0
                    best_price_since_entry = entry_price

                    # Partial close trigger price
                    if self.stop_loss_mode == "fixed":
                        partial_dist = self.partial_close_fixed_points
                    else:
                        partial_dist = entry_atr * self.partial_close_atr_mult
                    if position == 1:
                        partial_close_trigger = entry_price + partial_dist
                    else:
                        partial_close_trigger = entry_price - partial_dist

                    trailing_stop_price = 0.0
            else:
                # In position - check for exit conditions
                bars_in_trade += 1
                current_high = high[i]
                current_low = low[i]
                current_close = close[i]
                exit_price = None
                exit_reason = ""

                # Update best price since entry (for trailing stop)
                if position == 1:
                    best_price_since_entry = max(best_price_since_entry, current_high)
                else:
                    best_price_since_entry = min(best_price_since_entry, current_low)

                if not partial_close_done:
                    # PHASE 1: Full position - check SL, partial close trigger, TP
                    if position == 1:  # Long position
                        if current_low <= stop_loss:
                            # Full stop loss hit - close entire position
                            exit_price = stop_loss
                            exit_reason = "stop_loss"
                        elif (
                            self.partial_close_enabled
                            and current_high >= partial_close_trigger
                        ):
                            # Partial close triggered - book 50% profit
                            partial_exit_price = partial_close_trigger
                            partial_pnl_gross = (partial_exit_price - entry_price) * self.partial_close_fraction

                            # Compute partial close cost
                            entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[i]
                            partial_close_cost = self.cost_model.partial_exit_cost(
                                entry_price=entry_price,
                                exit_price=partial_exit_price,
                                direction=position,
                                entry_volatility=entry_vol,
                                exit_volatility=tr_arr[i],
                                avg_volatility=avg_volatility[i],
                                fraction=self.partial_close_fraction,
                            )

                            partial_close_pnl = partial_pnl_gross
                            partial_close_done = True

                            # Reset best price from partial close point for trailing
                            best_price_since_entry = current_high

                            # Activate trailing stop
                            if self.stop_loss_mode == "fixed":
                                trailing_dist = self.trailing_stop_fixed_points
                            else:
                                trailing_dist = entry_atr * self.trailing_stop_atr_mult
                            trailing_stop_price = best_price_since_entry - trailing_dist

                            # Also check if TP was hit on the same bar
                            if current_high >= take_profit:
                                exit_price = take_profit
                                exit_reason = "take_profit"

                        elif current_high >= take_profit:
                            # Take profit hit before partial close - exit full position
                            exit_price = take_profit
                            exit_reason = "take_profit"

                    else:  # Short position
                        if current_high >= stop_loss:
                            # Full stop loss hit
                            exit_price = stop_loss
                            exit_reason = "stop_loss"
                        elif (
                            self.partial_close_enabled
                            and current_low <= partial_close_trigger
                        ):
                            # Partial close triggered
                            partial_exit_price = partial_close_trigger
                            partial_pnl_gross = (entry_price - partial_exit_price) * self.partial_close_fraction

                            entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[i]
                            partial_close_cost = self.cost_model.partial_exit_cost(
                                entry_price=entry_price,
                                exit_price=partial_exit_price,
                                direction=position,
                                entry_volatility=entry_vol,
                                exit_volatility=tr_arr[i],
                                avg_volatility=avg_volatility[i],
                                fraction=self.partial_close_fraction,
                            )

                            partial_close_pnl = partial_pnl_gross
                            partial_close_done = True

                            # Reset best price from partial close point for trailing
                            best_price_since_entry = current_low

                            # Activate trailing stop
                            if self.stop_loss_mode == "fixed":
                                trailing_dist = self.trailing_stop_fixed_points
                            else:
                                trailing_dist = entry_atr * self.trailing_stop_atr_mult
                            trailing_stop_price = best_price_since_entry + trailing_dist

                            # Also check if TP hit on same bar
                            if current_low <= take_profit:
                                exit_price = take_profit
                                exit_reason = "take_profit"

                        elif current_low <= take_profit:
                            # Take profit hit before partial close
                            exit_price = take_profit
                            exit_reason = "take_profit"

                else:
                    # PHASE 2: Partial position (50% remaining) - trailing stop active
                    # Update trailing stop (only moves in profitable direction)
                    if self.stop_loss_mode == "fixed":
                        trailing_dist = self.trailing_stop_fixed_points
                    else:
                        trailing_dist = entry_atr * self.trailing_stop_atr_mult

                    if position == 1:
                        new_trailing = best_price_since_entry - trailing_dist
                        trailing_stop_price = max(trailing_stop_price, new_trailing)

                        # Check trailing stop hit
                        if current_low <= trailing_stop_price:
                            exit_price = trailing_stop_price
                            exit_reason = "trailing_stop"
                        elif current_high >= take_profit:
                            exit_price = take_profit
                            exit_reason = "take_profit"
                        elif current_low <= stop_loss:
                            # Original SL still protects remaining position
                            exit_price = stop_loss
                            exit_reason = "partial_then_stop"
                    else:
                        new_trailing = best_price_since_entry + trailing_dist
                        trailing_stop_price = min(trailing_stop_price, new_trailing)

                        # Check trailing stop hit
                        if current_high >= trailing_stop_price:
                            exit_price = trailing_stop_price
                            exit_reason = "trailing_stop"
                        elif current_low <= take_profit:
                            exit_price = take_profit
                            exit_reason = "take_profit"
                        elif current_high >= stop_loss:
                            # Original SL still protects remaining position
                            exit_price = stop_loss
                            exit_reason = "partial_then_stop"

                if exit_price is not None:
                    # Calculate P&L for the exit
                    if partial_close_done and exit_reason != "stop_loss":
                        # Partial was already booked - this exit is for the remaining fraction
                        remaining_fraction = 1.0 - self.partial_close_fraction
                        remaining_pnl_gross = (exit_price - entry_price) * position * remaining_fraction

                        # Cost for remaining exit
                        entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[i]
                        remaining_cost = self.cost_model.partial_exit_cost(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=position,
                            entry_volatility=entry_vol,
                            exit_volatility=tr_arr[i],
                            avg_volatility=avg_volatility[i],
                            fraction=remaining_fraction,
                        )

                        # Total P&L
                        pnl_gross = partial_close_pnl + remaining_pnl_gross
                        total_cost = partial_close_cost + remaining_cost
                        pnl_net = pnl_gross - total_cost
                        trailing_exit_pnl_val = remaining_pnl_gross

                    elif exit_reason == "stop_loss" and not partial_close_done:
                        # Full stop loss - entire position closed at loss
                        pnl_gross = (exit_price - entry_price) * position

                        entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[i]
                        total_cost = self.cost_model.total_cost_per_trade(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=position,
                            entry_volatility=entry_vol,
                            exit_volatility=tr_arr[i],
                            avg_volatility=avg_volatility[i],
                        )
                        pnl_net = pnl_gross - total_cost
                        partial_close_pnl = 0.0
                        trailing_exit_pnl_val = 0.0

                    elif exit_reason == "stop_loss" and partial_close_done:
                        # Should not happen (handled as partial_then_stop above)
                        # but handle gracefully
                        remaining_fraction = 1.0 - self.partial_close_fraction
                        remaining_pnl_gross = (exit_price - entry_price) * position * remaining_fraction

                        entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[i]
                        remaining_cost = self.cost_model.partial_exit_cost(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=position,
                            entry_volatility=entry_vol,
                            exit_volatility=tr_arr[i],
                            avg_volatility=avg_volatility[i],
                            fraction=remaining_fraction,
                        )

                        pnl_gross = partial_close_pnl + remaining_pnl_gross
                        total_cost = partial_close_cost + remaining_cost
                        pnl_net = pnl_gross - total_cost
                        trailing_exit_pnl_val = remaining_pnl_gross

                    else:
                        # Full exit without partial close (TP hit before partial trigger)
                        pnl_gross = (exit_price - entry_price) * position

                        entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[i]
                        total_cost = self.cost_model.total_cost_per_trade(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=position,
                            entry_volatility=entry_vol,
                            exit_volatility=tr_arr[i],
                            avg_volatility=avg_volatility[i],
                        )
                        pnl_net = pnl_gross - total_cost
                        partial_close_pnl = 0.0
                        trailing_exit_pnl_val = 0.0

                    # Record trade
                    entry_time = (
                        df.index[entry_idx]
                        if dates_arr is not None
                        else None
                    )
                    exit_time = (
                        df.index[i] if dates_arr is not None else None
                    )

                    trades.append(
                        Trade(
                            entry_idx=entry_idx,
                            exit_idx=i,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=position,
                            pnl_gross=pnl_gross,
                            pnl_net=pnl_net,
                            cost=total_cost,
                            entry_time=entry_time,
                            exit_time=exit_time,
                            exit_reason=exit_reason,
                            partial_close_pnl=partial_close_pnl,
                            trailing_exit_pnl=trailing_exit_pnl_val,
                        )
                    )

                    cumulative_gross += pnl_gross
                    cumulative_net += pnl_net

                    # Track daily trade count (count entry day)
                    entry_date = dates_arr[entry_idx] if dates_arr is not None else None
                    if entry_date is not None:
                        daily_trade_count[entry_date] = daily_trade_count.get(entry_date, 0) + 1

                    # Reset position
                    position = 0
                    entry_price = 0.0
                    bars_in_trade = 0
                    partial_close_done = False
                    partial_close_pnl = 0.0
                    partial_close_cost = 0.0

                    # If signal reversal would normally enter, check for new entry
                    if current_signal in (1, -1) and current_signal != 0:
                        new_trades_today = daily_trade_count.get(current_date, 0) if current_date else 0
                        new_limit_reached = (
                            self.max_trades_per_day is not None
                            and new_trades_today >= self.max_trades_per_day
                        )
                        if in_session and not new_limit_reached and exit_reason in ("stop_loss", "trailing_stop", "partial_then_stop"):
                            # Only allow re-entry on explicit signal after exit
                            pass  # Don't auto-enter on exit bar

            equity_gross_arr[i] = cumulative_gross
            equity_net_arr[i] = cumulative_net

        # Forced close of any position still open on the final bar. Used for
        # block-based splits so a trade can never straddle a block boundary.
        if self.close_at_end and position != 0 and n > 0:
            last = n - 1
            exit_price = close[last]
            remaining_fraction = (
                1.0 - self.partial_close_fraction if partial_close_done else 1.0
            )
            remaining_pnl_gross = (
                (exit_price - entry_price) * position * remaining_fraction
            )
            entry_vol = tr_arr[entry_idx] if entry_idx < n else avg_volatility[last]
            remaining_cost = self.cost_model.partial_exit_cost(
                entry_price=entry_price,
                exit_price=exit_price,
                direction=position,
                entry_volatility=entry_vol,
                exit_volatility=tr_arr[last],
                avg_volatility=avg_volatility[last],
                fraction=remaining_fraction,
            )
            pnl_gross = partial_close_pnl + remaining_pnl_gross
            total_cost = partial_close_cost + remaining_cost
            pnl_net = pnl_gross - total_cost

            trades.append(
                Trade(
                    entry_idx=entry_idx,
                    exit_idx=last,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=position,
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    cost=total_cost,
                    entry_time=df.index[entry_idx] if dates_arr is not None else None,
                    exit_time=df.index[last] if dates_arr is not None else None,
                    exit_reason="block_end",
                    partial_close_pnl=partial_close_pnl if partial_close_done else 0.0,
                    trailing_exit_pnl=remaining_pnl_gross if partial_close_done else 0.0,
                )
            )

            cumulative_gross += pnl_gross
            cumulative_net += pnl_net
            equity_gross_arr[last] = cumulative_gross
            equity_net_arr[last] = cumulative_net

        equity_gross = pd.Series(equity_gross_arr, index=df.index)
        equity_net = pd.Series(equity_net_arr, index=df.index)

        return trades, equity_gross, equity_net

    def _optimize_params(self, train_df: pd.DataFrame) -> dict[str, Any]:
        """Optimize strategy parameters via grid search on training data.

        Uses average net P&L per trade as the scoring metric (bounded,
        comparable across parameter sets). Requires a minimum of 5 trades
        to prevent degenerate single-trade parameter sets from winning.

        For large datasets (>3000 bars), uses a representative subsample
        to keep optimization time reasonable.

        Args:
            train_df: Training period DataFrame.

        Returns:
            Best parameters dict.
        """
        param_ranges = self.strategy.get_param_ranges()
        if not param_ranges:
            return {}

        # For large datasets, use a subsample for optimization speed
        # Take the last N bars (most recent = most relevant)
        max_opt_bars = 3000
        if len(train_df) > max_opt_bars:
            opt_df = train_df.iloc[-max_opt_bars:].copy()
        else:
            opt_df = train_df

        # Minimum trade count to consider a parameter set valid.
        min_trade_count = 5

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        original_params = self.strategy.params.copy()

        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        for combo in product(*param_values):
            # Set parameters
            test_params = dict(zip(param_names, combo))
            self.strategy.params.update(test_params)

            # Run backtest on training data (or subsample)
            try:
                signals = self.strategy.generate_signals(opt_df)
                trades, eq_gross, eq_net = self._simulate(
                    opt_df, signals["signal"]
                )

                # Gate: require minimum number of trades for a valid score
                if len(trades) < min_trade_count:
                    score = 0.0
                else:
                    # Use average net P&L per trade as a bounded metric.
                    score = sum(t.pnl_net for t in trades) / len(trades)

                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
            except Exception:
                continue

        # Restore original params (will be updated by caller)
        self.strategy.params = original_params

        return best_params
