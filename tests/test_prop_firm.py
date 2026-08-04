"""Tests for the FundedNext 50K prop firm sizing analysis.

Covers the one-stop-out elimination filter, the in-loop hard fails (daily
loss limit and equity floor), and the arithmetic ceiling arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.prop_firm import (
    PropFirmRules,
    arithmetic_ceiling_pct,
    evaluate_sizes,
    simulate_year,
    trades_per_day_counts,
)
from src.backtester.engine import Trade


def make_trades(pnls: list[float], per_day: int = 1) -> list[Trade]:
    """Build Trade records with the given net P&L in points."""
    trades = []
    day = pd.Timestamp("2024-01-02 10:00")
    for i, pnl in enumerate(pnls):
        entry = day + pd.Timedelta(days=i // per_day, minutes=5 * (i % per_day))
        trades.append(
            Trade(
                entry_idx=i,
                exit_idx=i + 1,
                entry_price=100.0,
                exit_price=100.0 + pnl,
                direction=1,
                pnl_gross=pnl,
                pnl_net=pnl,
                cost=0.0,
                entry_time=entry,
                exit_time=entry + pd.Timedelta(minutes=5),
            )
        )
    return trades


class TestTradesPerDayCounts:
    """Tests for the observed entries-per-day distribution."""

    def test_counts_trades_per_day(self):
        """Two trades on each of three days gives three counts of two."""
        trades = make_trades([1.0] * 6, per_day=2)
        counts = trades_per_day_counts(trades, trading_days=3)

        assert sorted(counts) == [2, 2, 2]

    def test_includes_zero_trade_days(self):
        """Trading days with no entries are counted as zero."""
        trades = make_trades([1.0, 1.0], per_day=1)
        counts = trades_per_day_counts(trades, trading_days=10)

        assert len(counts) == 10
        assert list(counts).count(0) == 8


class TestEliminationFilter:
    """Tests for the one-stop-out daily loss limit filter."""

    def test_micro_sizes_above_limit_are_eliminated(self):
        """With a 20-point stop, 26+ Micro breach the $1,000 daily limit."""
        rules = PropFirmRules()
        results = evaluate_sizes(
            trades=make_trades([5.0] * 20),
            trading_days=20,
            stop_points=20,
            rules=rules,
            n_sims=50,
        )
        micro = {r.contracts: r for r in results if r.contract_type == "Micro"}

        assert micro[25].stop_out_dollars == pytest.approx(1000.0)
        assert not micro[25].eliminated
        assert micro[26].stop_out_dollars == pytest.approx(1040.0)
        assert micro[26].eliminated
        assert micro[40].eliminated
        assert "daily loss limit" in micro[40].elimination_reason

    def test_mini_sizes_above_limit_are_eliminated(self):
        """With a 20-point stop, 3+ Mini breach the $1,000 daily limit."""
        results = evaluate_sizes(
            trades=make_trades([5.0] * 20),
            trading_days=20,
            stop_points=20,
            rules=PropFirmRules(),
            n_sims=50,
        )
        mini = {r.contracts: r for r in results if r.contract_type == "Mini"}

        assert not mini[2].eliminated
        assert mini[3].eliminated
        assert mini[4].eliminated

    def test_all_sizes_covered(self):
        """Every size from 1-40 Micro and 1-4 Mini is reported."""
        results = evaluate_sizes(
            trades=make_trades([5.0] * 20),
            trading_days=20,
            stop_points=30,
            rules=PropFirmRules(),
            n_sims=20,
        )

        assert len([r for r in results if r.contract_type == "Micro"]) == 40
        assert len([r for r in results if r.contract_type == "Mini"]) == 4


class TestSimulateYear:
    """Tests for the trade-by-trade year simulation."""

    def test_steady_winner_survives_and_compounds_linearly(self):
        """One +5 point trade per day at $2/pt gives +$2,500 over 250 days."""
        rules = PropFirmRules()
        equity, alive, reached = simulate_year(
            trade_pnl_points=np.array([5.0]),
            day_counts=np.array([1]),
            dollars_per_point=2.0,
            rules=rules,
            n_sims=100,
            seed=1,
        )

        assert alive.all()
        assert equity == pytest.approx(52_500.0)
        assert not reached.any()  # 2,500 never reaches the 3,000 target

    def test_profit_target_flag_set_when_reached(self):
        """The target flag trips once equity clears account + target."""
        rules = PropFirmRules()
        _, alive, reached = simulate_year(
            trade_pnl_points=np.array([10.0]),
            day_counts=np.array([1]),
            dollars_per_point=2.0,
            rules=rules,
            n_sims=50,
            seed=1,
        )

        assert alive.all()
        assert reached.all()

    def test_equity_floor_is_a_hard_fail(self):
        """Grinding down to the floor kills the account mid-year."""
        rules = PropFirmRules()
        equity, alive, _ = simulate_year(
            trade_pnl_points=np.array([-20.0]),
            day_counts=np.array([1]),
            dollars_per_point=2.0,  # -$40 per day
            rules=rules,
            n_sims=50,
            seed=1,
        )

        assert not alive.any()
        # Dies at the first equity below 48,000, not at the end of the year.
        assert equity.min() >= 48_000.0 - 40.0
        assert equity.max() < 48_000.0

    def test_daily_loss_limit_is_a_hard_fail(self):
        """Two stop-outs in one day breach the daily limit and end the year."""
        rules = PropFirmRules()
        equity, alive, _ = simulate_year(
            trade_pnl_points=np.array([-20.0]),
            day_counts=np.array([2]),
            dollars_per_point=30.0,  # one stop = -$600, two = -$1,200
            rules=rules,
            n_sims=50,
            seed=1,
        )

        assert not alive.any()
        assert equity == pytest.approx(48_800.0)  # exactly two stop-outs

    def test_dead_account_stops_trading(self):
        """A busted simulation never trades again after the breach."""
        rules = PropFirmRules(trading_days_per_year=250)
        equity, alive, _ = simulate_year(
            trade_pnl_points=np.array([-20.0]),
            day_counts=np.array([2]),
            dollars_per_point=30.0,
            rules=rules,
            n_sims=20,
            seed=2,
        )

        assert not alive.any()
        # Only the two trades of the first day were taken.
        assert equity == pytest.approx(48_800.0)

    def test_no_trades_means_no_change(self):
        """An empty trade list leaves the account untouched."""
        rules = PropFirmRules()
        equity, alive, reached = simulate_year(
            trade_pnl_points=np.array([]),
            day_counts=np.array([1]),
            dollars_per_point=2.0,
            rules=rules,
            n_sims=10,
            seed=1,
        )

        assert equity == pytest.approx(50_000.0)
        assert alive.all()
        assert not reached.any()


class TestArithmeticCeiling:
    """Tests for the contract-cap arithmetic ceiling."""

    def test_ceiling_matches_hand_calculation(self):
        """250 days x 2 trades x 1 point x $80/pt = $40,000 = 80% of 50K."""
        rules = PropFirmRules()
        ceiling = arithmetic_ceiling_pct(
            ev_per_trade_points=1.0, trades_per_day=2.0, rules=rules
        )

        assert ceiling == pytest.approx(80.0)

    def test_ceiling_scales_with_ev(self):
        """Doubling EV per trade doubles the ceiling."""
        rules = PropFirmRules()
        low = arithmetic_ceiling_pct(1.0, 1.5, rules)
        high = arithmetic_ceiling_pct(2.0, 1.5, rules)

        assert high == pytest.approx(low * 2)

    def test_negative_ev_gives_negative_ceiling(self):
        """A losing strategy has no positive ceiling."""
        assert arithmetic_ceiling_pct(-1.0, 2.0, PropFirmRules()) < 0
