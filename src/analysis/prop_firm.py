"""Prop firm sizing analysis under FundedNext 50K rules.

Answers one question honestly: what is the maximum achievable return on a
FundedNext 50K account with this strategy, given the contract cap and the
daily loss limit?

Two independent pieces:

1. A hard filter. One stop-out costs `stop_points x $/point x contracts`. Any
   size where a single stop-out alone breaches the $1,000 daily loss limit is
   eliminated before any simulation runs.
2. A Monte Carlo over a 250-trading-day year for every surviving size. Trades
   are resampled from the true OOS trade list (each P&L produced by the
   bar-by-bar backtest) and replayed trade-by-trade. The daily loss limit and
   the $48,000 equity floor are hard fails checked inside the loop: when one
   trips, the account is dead and the realized equity at that instant is the
   final equity. Nothing is capped after the fact.
"""

from dataclasses import dataclass

import numpy as np

from src.backtester.engine import Trade


@dataclass
class PropFirmRules:
    """FundedNext 50K account rules."""

    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    daily_loss_limit: float = 1_000.0
    equity_floor: float = 48_000.0
    micro_point_value: float = 2.0
    mini_point_value: float = 20.0
    max_micro_contracts: int = 40
    max_mini_contracts: int = 4
    trading_days_per_year: int = 250

    @classmethod
    def from_config(cls, config: dict) -> "PropFirmRules":
        """Build rules from the 'prop_firm' config section."""
        cfg = config.get("prop_firm", {})
        return cls(
            account_size=float(cfg.get("account_size", 50_000)),
            profit_target=float(cfg.get("profit_target", 3_000)),
            daily_loss_limit=float(cfg.get("daily_loss_limit", 1_000)),
            equity_floor=float(cfg.get("equity_floor", 48_000)),
            micro_point_value=float(cfg.get("micro_point_value", 2.0)),
            mini_point_value=float(cfg.get("mini_point_value", 20.0)),
            max_micro_contracts=int(cfg.get("max_micro_contracts", 40)),
            max_mini_contracts=int(cfg.get("max_mini_contracts", 4)),
            trading_days_per_year=int(cfg.get("trading_days_per_year", 250)),
        )


@dataclass
class SizeResult:
    """Outcome for one contract size."""

    label: str
    contracts: int
    contract_type: str
    dollars_per_point: float
    stop_out_dollars: float
    eliminated: bool
    elimination_reason: str = ""
    p_survive: float = 0.0
    p_reach_target: float = 0.0
    median_return_pct: float = 0.0
    p95_return_pct: float = 0.0
    max_return_pct: float = 0.0


def trades_per_day_counts(trades: list[Trade], trading_days: int) -> np.ndarray:
    """Observed number of entries per trading day, including zero-trade days.

    Args:
        trades: Trades from the evaluated period.
        trading_days: Number of trading days in that period.

    Returns:
        Array of length trading_days with the entry count for each day.
    """
    per_day: dict = {}
    for t in trades:
        if t.entry_time is None:
            continue
        day = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
        per_day[day] = per_day.get(day, 0) + 1

    counts = list(per_day.values())
    zero_days = max(0, trading_days - len(counts))
    counts.extend([0] * zero_days)

    if not counts:
        return np.zeros(1, dtype=int)
    return np.asarray(counts, dtype=int)


def simulate_year(
    trade_pnl_points: np.ndarray,
    day_counts: np.ndarray,
    dollars_per_point: float,
    rules: PropFirmRules,
    n_sims: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay a trading year trade-by-trade with hard in-loop rule checks.

    Vectorized across simulations, sequential across days and across the
    trades inside each day. A simulation that breaches the daily loss limit or
    the equity floor stops trading immediately and keeps the equity it had at
    that moment.

    Args:
        trade_pnl_points: Net P&L per trade in points (from the backtest).
        day_counts: Observed entries-per-day distribution to resample.
        dollars_per_point: Dollar value of one point at this contract size.
        rules: Account rules.
        n_sims: Number of simulated years.
        seed: RNG seed.

    Returns:
        Tuple of (final equity per sim, survived flag per sim,
        reached-profit-target flag per sim).
    """
    rng = np.random.default_rng(seed)

    equity = np.full(n_sims, rules.account_size, dtype=float)
    alive = np.ones(n_sims, dtype=bool)
    reached = np.zeros(n_sims, dtype=bool)
    target_equity = rules.account_size + rules.profit_target

    max_trades_in_day = int(day_counts.max()) if len(day_counts) else 0
    n_trades = len(trade_pnl_points)
    if n_trades == 0 or max_trades_in_day == 0:
        return equity, alive, reached

    for _ in range(rules.trading_days_per_year):
        if not alive.any():
            break

        todays_trades = day_counts[rng.integers(0, len(day_counts), size=n_sims)]
        day_pnl = np.zeros(n_sims, dtype=float)

        for slot in range(max_trades_in_day):
            active = alive & (todays_trades > slot)
            if not active.any():
                continue

            pnl = trade_pnl_points[rng.integers(0, n_trades, size=n_sims)] * dollars_per_point
            equity[active] += pnl[active]
            day_pnl[active] += pnl[active]

            reached |= active & (equity >= target_equity)

            # Hard fails, checked immediately after each trade. The boundary
            # case (a loss exactly equal to the limit) counts as surviving,
            # consistent with the "exceeds the limit" elimination filter.
            breached = active & (
                (day_pnl < -rules.daily_loss_limit) | (equity < rules.equity_floor)
            )
            alive &= ~breached

    return equity, alive, reached


def evaluate_sizes(
    trades: list[Trade],
    trading_days: int,
    stop_points: float,
    rules: PropFirmRules,
    n_sims: int = 10_000,
    seed: int = 42,
) -> list[SizeResult]:
    """Evaluate every contract size from 1-40 Micro and 1-4 Mini.

    Args:
        trades: OOS trades (net P&L in points) to resample from.
        trading_days: Trading days observed in the OOS period.
        stop_points: Locked stop distance in points.
        rules: Account rules.
        n_sims: Monte Carlo paths per surviving size.
        seed: Base RNG seed.

    Returns:
        List of SizeResult, Micro sizes first then Mini.
    """
    pnl_points = np.array([t.pnl_net for t in trades], dtype=float)
    day_counts = trades_per_day_counts(trades, trading_days)

    contract_specs = [
        ("Micro", rules.micro_point_value, rules.max_micro_contracts),
        ("Mini", rules.mini_point_value, rules.max_mini_contracts),
    ]

    results: list[SizeResult] = []
    for contract_type, point_value, max_contracts in contract_specs:
        for contracts in range(1, max_contracts + 1):
            dollars_per_point = point_value * contracts
            stop_out_dollars = stop_points * dollars_per_point
            label = f"{contracts} {contract_type}"

            if stop_out_dollars > rules.daily_loss_limit:
                results.append(
                    SizeResult(
                        label=label,
                        contracts=contracts,
                        contract_type=contract_type,
                        dollars_per_point=dollars_per_point,
                        stop_out_dollars=stop_out_dollars,
                        eliminated=True,
                        elimination_reason=(
                            f"one stop-out = ${stop_out_dollars:,.0f} > "
                            f"${rules.daily_loss_limit:,.0f} daily loss limit"
                        ),
                    )
                )
                continue

            equity, alive, reached = simulate_year(
                trade_pnl_points=pnl_points,
                day_counts=day_counts,
                dollars_per_point=dollars_per_point,
                rules=rules,
                n_sims=n_sims,
                seed=seed + contracts + (1000 if contract_type == "Mini" else 0),
            )
            returns_pct = (equity - rules.account_size) / rules.account_size * 100.0

            results.append(
                SizeResult(
                    label=label,
                    contracts=contracts,
                    contract_type=contract_type,
                    dollars_per_point=dollars_per_point,
                    stop_out_dollars=stop_out_dollars,
                    eliminated=False,
                    p_survive=float(alive.mean()),
                    p_reach_target=float(reached.mean()),
                    median_return_pct=float(np.median(returns_pct)),
                    p95_return_pct=float(np.percentile(returns_pct, 95)),
                    max_return_pct=float(returns_pct.max()),
                )
            )

    return results


def arithmetic_ceiling_pct(
    ev_per_trade_points: float,
    trades_per_day: float,
    rules: PropFirmRules,
) -> float:
    """Annual return % if the contract cap were run flat out with no bad days.

    Assumes: maximum allowed size (40 Micro) on every trade, the observed
    trades/day, the OOS EV per trade earned on every single trade, no losing
    days, no daily-limit breaches and no compounding (the contract cap does
    not rise with equity, so the accumulation is linear).

    Args:
        ev_per_trade_points: OOS expected value per trade, in points.
        trades_per_day: Observed trades per trading day.
        rules: Account rules.

    Returns:
        Annual return as a percentage of the account size.
    """
    dollars_per_point = rules.micro_point_value * rules.max_micro_contracts
    annual_dollars = (
        rules.trading_days_per_year
        * trades_per_day
        * ev_per_trade_points
        * dollars_per_point
    )
    return annual_dollars / rules.account_size * 100.0
