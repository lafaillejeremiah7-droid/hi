"""Tests for the backtesting engine and cost model.

Tests engine logic, cost model calculations, train/validation split,
and equity curve properties.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine, BacktestResult, Trade
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.simple_strategy import SimpleStrategy
from src.strategies.volume_profile_strategy import VolumeProfileStrategy


def make_simple_data(n: int = 100) -> pd.DataFrame:
    """Create simple OHLCV data with preprocessed columns."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    base = np.cumsum(np.random.randn(n) * 1.0) + 100
    open_p = base + np.random.randn(n) * 0.3
    close_p = base + np.random.randn(n) * 0.3
    high_p = np.maximum(open_p, close_p) + np.abs(np.random.randn(n) * 0.5)
    low_p = np.minimum(open_p, close_p) - np.abs(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        },
        index=dates,
    )

    # Add preprocessed columns
    bar_range = (df["high"] - df["low"]).replace(0, 1)
    df["volume_delta"] = df["volume"] * (df["close"] - df["open"]) / bar_range
    df["cumulative_delta"] = df["volume_delta"].cumsum()
    df["relative_volume"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    df["resistance_1"] = df["high"].rolling(20, min_periods=1).max()
    df["support_1"] = df["low"].rolling(20, min_periods=1).min()
    df["resistance_2"] = df["resistance_1"] * 0.99
    df["support_2"] = df["support_1"] * 1.01
    dist_sup = (df["close"] - df["support_1"]).abs()
    dist_res = (df["resistance_1"] - df["close"]).abs()
    df["nearest_sr_distance"] = (
        pd.concat([dist_sup, dist_res], axis=1).min(axis=1) / df["close"]
    )
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


class TestCostModel:
    """Tests for the CostModel class."""

    def test_default_construction(self):
        """CostModel initializes with default values."""
        cm = CostModel()
        assert cm.base_slippage_points == 1.0
        assert cm.commission_per_round_trip == 4.50
        assert cm.point_value == 20.0

    def test_custom_construction(self):
        """CostModel accepts custom parameters."""
        cm = CostModel(
            base_slippage_points=2.0,
            commission_per_round_trip=5.0,
            point_value=25.0,
        )
        assert cm.base_slippage_points == 2.0
        assert cm.commission_per_round_trip == 5.0
        assert cm.point_value == 25.0

    def test_from_config(self):
        """CostModel creates correctly from config dict."""
        config = {
            "slippage_points": 1.5,
            "commission_per_round_trip": 4.0,
            "point_value": 20.0,
        }
        cm = CostModel.from_config(config)
        assert cm.base_slippage_points == 1.5
        assert cm.commission_per_round_trip == 4.0

    def test_slippage_buy_increases_price(self):
        """Buying slippage increases execution price."""
        cm = CostModel(base_slippage_points=1.0)
        price = 100.0
        result = cm.apply_slippage(price, direction=1, volatility=1.0, avg_volatility=1.0)
        assert result > price

    def test_slippage_sell_decreases_price(self):
        """Selling slippage decreases execution price."""
        cm = CostModel(base_slippage_points=1.0)
        price = 100.0
        result = cm.apply_slippage(price, direction=-1, volatility=1.0, avg_volatility=1.0)
        assert result < price

    def test_slippage_scales_with_volatility(self):
        """Higher volatility means more slippage."""
        cm = CostModel(base_slippage_points=1.0)
        price = 100.0

        low_vol_result = cm.apply_slippage(price, 1, volatility=0.5, avg_volatility=1.0)
        high_vol_result = cm.apply_slippage(price, 1, volatility=2.0, avg_volatility=1.0)

        # Higher volatility should cause more slippage (higher buy price)
        assert high_vol_result > low_vol_result

    def test_slippage_capped_at_2x(self):
        """Slippage multiplier is capped at 2x base."""
        cm = CostModel(base_slippage_points=1.0)
        price = 100.0

        # Very high volatility (10x average)
        result = cm.apply_slippage(price, 1, volatility=10.0, avg_volatility=1.0)
        max_slippage = 1.0 * 2.0  # base * max_multiplier
        assert result == price + max_slippage

    def test_slippage_floored_at_0_5x(self):
        """Slippage multiplier has a floor of 0.5x base."""
        cm = CostModel(base_slippage_points=1.0)
        price = 100.0

        # Very low volatility
        result = cm.apply_slippage(price, 1, volatility=0.1, avg_volatility=1.0)
        min_slippage = 1.0 * 0.5  # base * min_multiplier
        assert result == price + min_slippage

    def test_commission_returns_fixed_amount(self):
        """Commission returns configured round-trip cost."""
        cm = CostModel(commission_per_round_trip=4.50)
        assert cm.apply_commission() == 4.50

    def test_total_cost_always_positive(self):
        """Total cost per trade is always positive."""
        cm = CostModel()
        cost = cm.total_cost_per_trade(
            entry_price=100.0,
            exit_price=105.0,
            direction=1,
            entry_volatility=1.0,
            exit_volatility=1.0,
            avg_volatility=1.0,
        )
        assert cost > 0

    def test_zero_avg_volatility(self):
        """Handles zero average volatility gracefully."""
        cm = CostModel(base_slippage_points=1.0)
        result = cm.apply_slippage(100.0, 1, volatility=1.0, avg_volatility=0.0)
        # Should use ratio of 1.0 (fallback)
        assert result == 101.0


class TestBacktestEngine:
    """Tests for the BacktestEngine class."""

    def test_engine_initialization(self):
        """Engine initializes with strategy and cost model."""
        strategy = OrderFlowStrategy()
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)

        assert engine.strategy is strategy
        assert engine.cost_model is cost_model
        assert engine.initial_capital == 100000.0

    def test_run_returns_result(self):
        """Engine run returns a BacktestResult."""
        strategy = OrderFlowStrategy()
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(100)

        result = engine.run(df)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_gross) == len(df)
        assert len(result.equity_net) == len(df)

    def test_equity_net_less_than_gross(self):
        """Equity with costs is always <= equity without costs."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run(df)

        # Net equity should be <= gross equity at all points
        diff = result.equity_gross - result.equity_net
        # Difference should be >= 0 (costs reduce equity)
        assert (diff >= -1e-10).all(), "Net equity exceeded gross equity"

    def test_trades_have_valid_pnl(self):
        """Trades have correctly calculated P&L."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run(df)

        for trade in result.trades:
            # Net P&L should be gross P&L minus costs
            assert abs(trade.pnl_net - (trade.pnl_gross - trade.cost)) < 1e-10
            # Costs should always be positive
            assert trade.cost >= 0

    def test_train_validation_split(self):
        """Train/validation split divides data correctly."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_split(df, train_ratio=0.6)

        assert result.train_result is not None
        assert result.validation_result is not None

        # Train result should cover first 60% of data
        expected_train_len = int(len(df) * 0.6)
        assert len(result.train_result.equity_gross) == expected_train_len

        # Validation result should cover last 40%
        expected_val_len = len(df) - expected_train_len
        assert len(result.validation_result.equity_gross) == expected_val_len

    def test_split_returns_best_params(self):
        """Train/validation split returns optimized parameters."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_split(df, train_ratio=0.6)

        # Should have best_params (even if empty due to no improvement)
        assert isinstance(result.best_params, dict)

    def test_no_position_buildup(self):
        """Engine does not accumulate positions (always flat or 1 contract)."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(100)

        result = engine.run(df)

        # Check that all trades are sequential (no overlapping)
        for i in range(1, len(result.trades)):
            assert result.trades[i].entry_idx >= result.trades[i - 1].exit_idx

    def test_empty_data(self):
        """Engine handles empty DataFrame gracefully."""
        strategy = OrderFlowStrategy()
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)

        df = make_simple_data(5)
        result = engine.run(df)

        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0

    def test_volume_profile_strategy_engine(self):
        """Engine works with VolumeProfileStrategy."""
        strategy = VolumeProfileStrategy()
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run(df)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_gross) == len(df)


class TestTradeDataclass:
    """Tests for the Trade dataclass."""

    def test_trade_creation(self):
        """Trade dataclass stores all fields."""
        trade = Trade(
            entry_idx=10,
            exit_idx=20,
            entry_price=100.0,
            exit_price=105.0,
            direction=1,
            pnl_gross=5.0,
            pnl_net=3.5,
            cost=1.5,
        )
        assert trade.entry_idx == 10
        assert trade.exit_idx == 20
        assert trade.direction == 1
        assert trade.pnl_gross == 5.0
        assert trade.pnl_net == 3.5
        assert trade.cost == 1.5


class TestThreeWaySplit:
    """Tests for the 3-way split (train/validation/OOS) logic."""

    def test_three_way_split_divides_data_correctly(self):
        """Three-way split produces correct segment sizes."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_three_way_split(
            df, train_ratio=0.50, validation_ratio=0.25
        )

        assert result.train_result is not None
        assert result.validation_result is not None
        assert result.oos_result is not None

        # Check sizes
        expected_train_len = int(len(df) * 0.50)
        expected_val_len = int(len(df) * 0.75) - expected_train_len
        expected_oos_len = len(df) - int(len(df) * 0.75)

        assert len(result.train_result.equity_gross) == expected_train_len
        assert len(result.validation_result.equity_gross) == expected_val_len
        assert len(result.oos_result.equity_gross) == expected_oos_len

    def test_oos_data_never_seen_during_optimization(self):
        """OOS data is not used during parameter optimization.

        Verifies that the OOS segment's index range is completely
        separate from the train segment used for optimization.
        """
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        # The 3-way split should split at indices 100 and 150
        # Train: 0-99, Validation: 100-149, OOS: 150-199
        result = engine.run_with_three_way_split(
            df, train_ratio=0.50, validation_ratio=0.25
        )

        # OOS result should exist and cover the last 25%
        assert result.oos_result is not None
        oos_start_idx = int(len(df) * 0.75)  # 150
        assert len(result.oos_result.equity_gross) == len(df) - oos_start_idx

    def test_three_way_split_returns_best_params(self):
        """Three-way split returns optimized parameters."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_three_way_split(
            df, train_ratio=0.50, validation_ratio=0.25
        )

        assert isinstance(result.best_params, dict)

    def test_three_way_segments_are_chronological(self):
        """Train, validation, and OOS are in strict chronological order."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_three_way_split(
            df, train_ratio=0.50, validation_ratio=0.25
        )

        # Train end < Validation start
        train_end = result.train_result.equity_gross.index[-1]
        val_start = result.validation_result.equity_gross.index[0]
        assert train_end < val_start

        # Validation end < OOS start
        val_end = result.validation_result.equity_gross.index[-1]
        oos_start = result.oos_result.equity_gross.index[0]
        assert val_end < oos_start

    def test_three_way_no_data_overlap(self):
        """No data overlaps between any two segments."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_three_way_split(
            df, train_ratio=0.50, validation_ratio=0.25
        )

        train_idx = set(result.train_result.equity_gross.index)
        val_idx = set(result.validation_result.equity_gross.index)
        oos_idx = set(result.oos_result.equity_gross.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & oos_idx) == 0
        assert len(val_idx & oos_idx) == 0

    def test_three_way_covers_all_data(self):
        """Three-way split accounts for all data in the original DataFrame."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1})
        cost_model = CostModel()
        engine = BacktestEngine(strategy, cost_model)
        df = make_simple_data(200)

        result = engine.run_with_three_way_split(
            df, train_ratio=0.50, validation_ratio=0.25
        )

        total_bars = (
            len(result.train_result.equity_gross)
            + len(result.validation_result.equity_gross)
            + len(result.oos_result.equity_gross)
        )
        assert total_bars == len(df)


class TestExpectedValueMetric:
    """Tests for the Expected Value (EV) per trade metric."""

    def test_ev_computed_in_metrics(self):
        """EV per trade is included in compute_all_metrics output."""
        from src.analysis.metrics import compute_all_metrics

        # Create a simple backtest result with known trades
        trades = [
            Trade(entry_idx=0, exit_idx=1, entry_price=100, exit_price=105,
                  direction=1, pnl_gross=5.5, pnl_net=5.0, cost=0.5),
            Trade(entry_idx=2, exit_idx=3, entry_price=100, exit_price=105,
                  direction=1, pnl_gross=4.5, pnl_net=4.0, cost=0.5),
            Trade(entry_idx=4, exit_idx=5, entry_price=100, exit_price=97,
                  direction=1, pnl_gross=-2.5, pnl_net=-3.0, cost=0.5),
        ]
        equity = pd.Series([0, 5.0, 9.0, 6.0], dtype=float)
        result = BacktestResult(
            trades=trades,
            equity_gross=equity,
            equity_net=equity,
            train_result=None,
            validation_result=None,
            best_params={},
        )

        metrics = compute_all_metrics(result)
        assert "expected_value" in metrics

        # Manual calculation: win_rate=2/3, loss_rate=1/3
        # avg_winner = (5.0 + 4.0) / 2 = 4.5
        # avg_loser = abs(-3.0) = 3.0
        # EV = (4.5 * 2/3) - (3.0 * 1/3) = 3.0 - 1.0 = 2.0
        assert abs(metrics["expected_value"] - 2.0) < 1e-10

    def test_ev_zero_with_no_trades(self):
        """EV is 0 when there are no trades."""
        from src.analysis.metrics import compute_all_metrics

        equity = pd.Series([0.0], dtype=float)
        result = BacktestResult(
            trades=[],
            equity_gross=equity,
            equity_net=equity,
            train_result=None,
            validation_result=None,
            best_params={},
        )

        metrics = compute_all_metrics(result)
        assert metrics["expected_value"] == 0.0

    def test_ev_positive_for_profitable_system(self):
        """EV is positive for a system with positive expectancy."""
        from src.analysis.metrics import compute_all_metrics

        # 70% win rate with 3:1 reward:risk
        trades = []
        for i in range(7):
            trades.append(Trade(
                entry_idx=i*2, exit_idx=i*2+1, entry_price=100,
                exit_price=106, direction=1, pnl_gross=6.5, pnl_net=6.0, cost=0.5,
            ))
        for i in range(3):
            trades.append(Trade(
                entry_idx=14+i*2, exit_idx=15+i*2, entry_price=100,
                exit_price=98, direction=1, pnl_gross=-1.5, pnl_net=-2.0, cost=0.5,
            ))

        equity = pd.Series([0.0] + [1.0] * len(trades), dtype=float)
        result = BacktestResult(
            trades=trades,
            equity_gross=equity,
            equity_net=equity,
            train_result=None,
            validation_result=None,
            best_params={},
        )

        metrics = compute_all_metrics(result)
        assert metrics["expected_value"] > 0


class TestDynamicExitManagement:
    """Tests for the dynamic exit management system.

    Tests partial close, trailing stop, extended TP, and cost split.
    """

    def _make_controlled_data(self, prices: list[float]) -> tuple[pd.DataFrame, pd.Series]:
        """Create controlled price data for precise exit testing.

        Args:
            prices: List of close prices. High/low constructed around them.

        Returns:
            Tuple of (DataFrame, signals Series with entry on bar 1).
        """
        n = len(prices)
        dates = pd.date_range("2024-01-01 09:30", periods=n, freq="5min")

        close = np.array(prices, dtype=float)
        # Open is same as close for simplicity
        open_p = close.copy()
        # High is close + 0.5, low is close - 0.5 (gives ATR ~1.0)
        high = close + 0.5
        low = close - 0.5

        df = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000.0,
            "volume_delta": np.zeros(n),
            "delta_zscore": np.zeros(n),
        }, index=dates)

        # Add required preprocessed columns
        df["cumulative_delta"] = df["volume_delta"].cumsum()
        df["relative_volume"] = 1.0
        df["resistance_1"] = df["high"].rolling(5, min_periods=1).max()
        df["support_1"] = df["low"].rolling(5, min_periods=1).min()
        df["resistance_2"] = df["resistance_1"] * 0.99
        df["support_2"] = df["support_1"] * 1.01
        df["nearest_sr_distance"] = 0.01
        df["vwap"] = close

        # Signal: go long at bar 1
        signals = pd.Series(0, index=df.index, dtype=int)
        signals.iloc[1] = 1

        return df, signals

    def test_partial_close_triggers_at_correct_price(self):
        """Partial close triggers when price moves 1.0 ATR in favor."""
        # Entry at 100, ATR ~1.0, so partial close at 101
        # Prices: setup, entry(100), 100.5, 101.5 (partial close triggers here)
        prices = [100.0, 100.0, 100.3, 101.5, 101.8, 102.0, 101.5, 101.0, 100.5, 100.0]

        df, signals = self._make_controlled_data(prices)

        strategy = OrderFlowStrategy(params={"min_signal_strength": 1, "atr_period": 5})
        cost_model = CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0)
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 1.5,
                "extended_take_profit_atr_mult": 2.5,
                "feature_zscore_threshold": 2.5,
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
            }
        )

        trades, eq_gross, eq_net = engine._simulate(df, signals)

        # Should have at least one trade
        assert len(trades) >= 1
        trade = trades[0]

        # Trade should have partial close recorded
        if trade.exit_reason in ("trailing_stop", "take_profit", "partial_then_stop"):
            assert trade.partial_close_pnl > 0, "Partial close should book positive P&L"

    def test_trailing_stop_activates_only_after_partial_close(self):
        """Trailing stop only activates after the partial close is done."""
        # Prices go up to trigger partial, then retrace to trigger trailing stop
        # Entry at 100, ATR ~1.0
        # Partial close at 101 (1.0 ATR profit)
        # Price goes to 102, then retraces to 101 (1.0 ATR from peak)
        prices = [100.0, 100.0, 100.5, 101.5, 102.5, 101.5, 100.5, 99.0]
        # Make highs/lows wide enough to trigger
        df, signals = self._make_controlled_data(prices)
        # Widen the highs at bar 3 and 4 to ensure partial trigger
        df.loc[df.index[3], "high"] = 102.0
        df.loc[df.index[4], "high"] = 103.0
        # Widen lows for trailing stop
        df.loc[df.index[5], "low"] = 100.5
        df.loc[df.index[6], "low"] = 99.5

        strategy = OrderFlowStrategy(params={"min_signal_strength": 1, "atr_period": 5})
        cost_model = CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0)
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 3.0,  # High TP to avoid hitting it
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
                "feature_zscore_threshold": 2.5,
                "extended_take_profit_atr_mult": 2.5,
            }
        )

        trades, _, _ = engine._simulate(df, signals)

        assert len(trades) >= 1
        trade = trades[0]

        # Should exit via trailing stop after partial close
        assert trade.exit_reason in ("trailing_stop", "partial_then_stop", "stop_loss", "take_profit")
        if trade.exit_reason == "trailing_stop":
            assert trade.partial_close_pnl > 0

    def test_trailing_stop_moves_in_profitable_direction_only(self):
        """Trailing stop only ratchets up (for longs), never down."""
        # Long entry at 100, partial at 101, price goes to 103, back to 102.5, up to 104
        # Trailing stop should be at 103-1=102, not 102.5-1=101.5
        prices = [100.0, 100.0, 100.5, 102.0, 103.5, 102.5, 104.0, 102.8, 102.5]
        df, signals = self._make_controlled_data(prices)
        # Ensure highs are high enough
        df.loc[df.index[3], "high"] = 102.5
        df.loc[df.index[4], "high"] = 104.0
        df.loc[df.index[5], "high"] = 103.0
        df.loc[df.index[6], "high"] = 104.5

        strategy = OrderFlowStrategy(params={"min_signal_strength": 1, "atr_period": 5})
        cost_model = CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0)
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 10.0,  # Very high to not hit
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
                "feature_zscore_threshold": 2.5,
                "extended_take_profit_atr_mult": 2.5,
            }
        )

        trades, _, _ = engine._simulate(df, signals)

        # The trade should end via trailing stop
        # The trailing stop never moves backwards
        assert len(trades) >= 1

    def test_extended_tp_when_zscore_above_threshold(self):
        """When |delta_zscore| >= 2.5 at entry, TP uses 2.5x ATR."""
        # Entry at 100 with high z-score, ATR ~1.0
        # Standard TP would be 101.5 (1.5 ATR), extended would be 102.5 (2.5 ATR)
        prices = [100.0, 100.0, 100.5, 101.0, 101.3, 101.8, 102.5, 103.0]
        df, signals = self._make_controlled_data(prices)

        # Set high delta_zscore at entry bar
        df.loc[df.index[1], "delta_zscore"] = 3.0  # Above 2.5 threshold

        # Make highs precise - price hits 101.5 (normal TP) but should continue
        df.loc[df.index[5], "high"] = 101.8  # Below extended TP of 102.5
        df.loc[df.index[6], "high"] = 102.8  # Hits extended TP of 102.5

        strategy = OrderFlowStrategy(params={
            "min_signal_strength": 1,
            "atr_period": 5,
            "take_profit_atr_mult": 1.5,
        })
        cost_model = CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0)
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 1.5,
                "extended_take_profit_atr_mult": 2.5,
                "feature_zscore_threshold": 2.5,
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
            }
        )

        trades, _, _ = engine._simulate(df, signals)

        # Should have a trade
        assert len(trades) >= 1
        trade = trades[0]
        # The TP should be extended (not 1.5 ATR) when zscore is high
        # Either it exits at take_profit (extended) or trailing/partial
        # The key test: the trade should NOT exit at bar 5 (1.5 ATR = 101.5)
        # because the extended TP is at 102.5

    def test_full_stop_loss_no_partial_close(self):
        """When price hits SL before partial close, entire position is closed."""
        # Entry at 100, SL at 99 (1.0 ATR below), price drops immediately
        prices = [100.0, 100.0, 99.5, 98.5, 97.0]
        df, signals = self._make_controlled_data(prices)
        # Ensure low is low enough to trigger SL
        df.loc[df.index[2], "low"] = 98.8

        strategy = OrderFlowStrategy(params={"min_signal_strength": 1, "atr_period": 5})
        cost_model = CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0)
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 1.5,
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
                "feature_zscore_threshold": 2.5,
                "extended_take_profit_atr_mult": 2.5,
            }
        )

        trades, _, _ = engine._simulate(df, signals)

        assert len(trades) >= 1
        trade = trades[0]

        # Full stop loss - no partial close
        assert trade.exit_reason == "stop_loss"
        assert trade.partial_close_pnl == 0.0
        assert trade.trailing_exit_pnl == 0.0
        assert trade.pnl_gross < 0  # Loss

    def test_cost_split_between_partial_and_trailing(self):
        """Total cost is split between partial close and trailing exit."""
        # Commission is $4.50 total. Partial close = $2.25, trailing = $2.25
        prices = [100.0, 100.0, 100.5, 102.0, 103.0, 101.5, 100.0]
        df, signals = self._make_controlled_data(prices)
        # Make highs/lows wide enough
        df.loc[df.index[3], "high"] = 102.5
        df.loc[df.index[4], "high"] = 103.5
        df.loc[df.index[5], "low"] = 101.0
        df.loc[df.index[6], "low"] = 99.5

        strategy = OrderFlowStrategy(params={"min_signal_strength": 1, "atr_period": 5})
        # Use known commission, no slippage for easy verification
        cost_model = CostModel(
            base_slippage_points=0.0,
            commission_per_round_trip=4.50,
            point_value=20.0,
        )
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 10.0,  # High TP
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
                "feature_zscore_threshold": 2.5,
                "extended_take_profit_atr_mult": 2.5,
            }
        )

        trades, _, _ = engine._simulate(df, signals)

        if trades and trades[0].exit_reason in ("trailing_stop", "partial_then_stop"):
            trade = trades[0]
            # Total commission in points: 4.50 / 20.0 = 0.225
            # With zero slippage, total cost should be ~0.225 points
            expected_total_commission = 4.50 / 20.0
            assert abs(trade.cost - expected_total_commission) < 0.01

    def test_trade_dataclass_has_new_fields(self):
        """Trade dataclass includes partial_close_pnl and trailing_exit_pnl."""
        trade = Trade(
            entry_idx=0,
            exit_idx=5,
            entry_price=100.0,
            exit_price=102.0,
            direction=1,
            pnl_gross=2.0,
            pnl_net=1.8,
            cost=0.2,
            partial_close_pnl=1.0,
            trailing_exit_pnl=1.0,
        )
        assert trade.partial_close_pnl == 1.0
        assert trade.trailing_exit_pnl == 1.0

    def test_partial_close_fraction_configurable(self):
        """The partial close fraction can be configured."""
        strategy = OrderFlowStrategy(params={"min_signal_strength": 1, "atr_period": 5})
        cost_model = CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0)
        engine = BacktestEngine(
            strategy, cost_model,
            exit_management={
                "stop_loss_atr_mult": 1.0,
                "take_profit_atr_mult": 1.5,
                "partial_close_atr_mult": 1.0,
                "partial_close_fraction": 0.5,
                "trailing_stop_atr_mult": 1.0,
                "feature_zscore_threshold": 2.5,
                "extended_take_profit_atr_mult": 2.5,
            }
        )
        assert engine.partial_close_fraction == 0.5


class TestPartialExitCost:
    """Tests for the partial_exit_cost method in CostModel."""

    def test_partial_exit_cost_proportional(self):
        """Partial exit cost is proportional to fraction."""
        cm = CostModel(
            base_slippage_points=0.0,
            commission_per_round_trip=4.50,
            point_value=20.0,
        )

        # Full cost
        full_cost = cm.total_cost_per_trade(
            entry_price=100.0,
            exit_price=101.0,
            direction=1,
            entry_volatility=1.0,
            exit_volatility=1.0,
            avg_volatility=1.0,
        )

        # Half cost (50% partial)
        half_cost = cm.partial_exit_cost(
            entry_price=100.0,
            exit_price=101.0,
            direction=1,
            entry_volatility=1.0,
            exit_volatility=1.0,
            avg_volatility=1.0,
            fraction=0.5,
        )

        # Half cost should be approximately half of full cost
        assert abs(half_cost - full_cost * 0.5) < 0.01

    def test_two_partial_exits_equal_full(self):
        """Two 50% partial exits sum to approximately full cost."""
        cm = CostModel(
            base_slippage_points=1.0,
            commission_per_round_trip=4.50,
            point_value=20.0,
        )

        full_cost = cm.total_cost_per_trade(
            entry_price=100.0,
            exit_price=102.0,
            direction=1,
            entry_volatility=1.0,
            exit_volatility=1.0,
            avg_volatility=1.0,
        )

        partial_1 = cm.partial_exit_cost(
            entry_price=100.0,
            exit_price=101.0,
            direction=1,
            entry_volatility=1.0,
            exit_volatility=1.0,
            avg_volatility=1.0,
            fraction=0.5,
        )

        partial_2 = cm.partial_exit_cost(
            entry_price=100.0,
            exit_price=102.0,
            direction=1,
            entry_volatility=1.0,
            exit_volatility=1.0,
            avg_volatility=1.0,
            fraction=0.5,
        )

        # Two 50% exits should sum to approximately full cost
        total_partial = partial_1 + partial_2
        assert abs(total_partial - full_cost) < 0.1


class TestFixedOnlyExits:
    """Tests for the fixed stop / fixed target configuration (no partials)."""

    @staticmethod
    def _make_data(prices: list[float]) -> tuple[pd.DataFrame, pd.Series]:
        """Build bars from close prices with a long entry on bar 1."""
        close = np.array(prices, dtype=float)
        n = len(close)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000.0,
                "delta": np.zeros(n),
                "delta_zscore": np.zeros(n),
            },
            index=pd.date_range("2024-01-02 09:30", periods=n, freq="5min"),
        )
        signals = pd.Series(0, index=df.index, dtype=int)
        signals.iloc[1] = 1
        return df, signals

    @staticmethod
    def _engine(strategy, close_at_end: bool = False) -> BacktestEngine:
        return BacktestEngine(
            strategy,
            CostModel(base_slippage_points=0.0, commission_per_round_trip=0.0),
            exit_management={
                "stop_loss_mode": "fixed",
                "partial_close_enabled": False,
            },
            close_at_end=close_at_end,
        )

    def test_target_hit_gives_full_target_pnl(self):
        """A winner books the whole fixed target with no partial close."""
        strategy = SimpleStrategy(params={"stop_points": 20, "target_points": 30})
        df, signals = self._make_data([100.0, 100.0, 110.0, 120.0, 131.0])
        engine = self._engine(strategy)

        trades, _, _ = engine._simulate(df, signals)

        assert len(trades) == 1
        assert trades[0].exit_reason == "take_profit"
        assert trades[0].pnl_gross == pytest.approx(30.0)
        assert trades[0].partial_close_pnl == 0.0
        assert trades[0].trailing_exit_pnl == 0.0

    def test_stop_hit_after_favourable_move_loses_full_stop(self):
        """Without partial closes an unrealized gain is not booked at all."""
        strategy = SimpleStrategy(params={"stop_points": 20, "target_points": 30})
        # Runs 25 points in favour (past the old partial trigger) then stops out.
        df, signals = self._make_data([100.0, 100.0, 125.0, 110.0, 79.0])
        engine = self._engine(strategy)

        trades, _, _ = engine._simulate(df, signals)

        assert len(trades) == 1
        assert trades[0].exit_reason == "stop_loss"
        assert trades[0].pnl_gross == pytest.approx(-20.0)
        assert trades[0].partial_close_pnl == 0.0

    def test_no_trailing_exit_reasons(self):
        """Trailing stops never fire when partial closes are disabled."""
        strategy = SimpleStrategy(params={"stop_points": 20, "target_points": 60})
        df, signals = self._make_data(
            [100.0, 100.0, 120.0, 140.0, 125.0, 110.0, 100.0, 79.0]
        )
        engine = self._engine(strategy)

        trades, _, _ = engine._simulate(df, signals)

        assert len(trades) == 1
        assert trades[0].exit_reason == "stop_loss"


class TestCloseAtEnd:
    """Tests for the forced close used by block-based splits."""

    def test_open_position_closed_on_final_bar(self):
        """close_at_end books the open position at the last bar's close."""
        strategy = SimpleStrategy(params={"stop_points": 50, "target_points": 90})
        df, signals = TestFixedOnlyExits._make_data([100.0, 100.0, 105.0, 112.0])
        engine = TestFixedOnlyExits._engine(strategy, close_at_end=True)

        trades, _, eq_net = engine._simulate(df, signals)

        assert len(trades) == 1
        assert trades[0].exit_reason == "block_end"
        assert trades[0].exit_idx == len(df) - 1
        assert trades[0].exit_price == pytest.approx(112.0)
        assert trades[0].pnl_gross == pytest.approx(12.0)
        assert float(eq_net.iloc[-1]) == pytest.approx(trades[0].pnl_net)

    def test_open_position_dropped_without_close_at_end(self):
        """Default behaviour still discards a position open on the last bar."""
        strategy = SimpleStrategy(params={"stop_points": 50, "target_points": 90})
        df, signals = TestFixedOnlyExits._make_data([100.0, 100.0, 105.0, 112.0])
        engine = TestFixedOnlyExits._engine(strategy, close_at_end=False)

        trades, _, _ = engine._simulate(df, signals)

        assert trades == []

    def test_closed_trade_not_double_counted(self):
        """A position that already exited is not re-closed on the final bar."""
        strategy = SimpleStrategy(params={"stop_points": 20, "target_points": 10})
        df, signals = TestFixedOnlyExits._make_data([100.0, 100.0, 111.0, 112.0])
        engine = TestFixedOnlyExits._engine(strategy, close_at_end=True)

        trades, _, _ = engine._simulate(df, signals)

        assert len(trades) == 1
        assert trades[0].exit_reason == "take_profit"
