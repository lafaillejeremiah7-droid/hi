"""Main entry point for the NAS100 backtesting framework.

Runs one simple strategy end to end on 5-minute NQ futures data with real
order flow:

1. Load configuration
2. Load 5-min data with real order flow (bid/ask volume, delta)
3. Preprocess
4. Filter to session hours (9:30-16:00 ET)
5. Build the interleaved-block train / validation / OOS split and print it
6. Select parameters on TRAIN, pick the best of the top 5 on VALIDATION, lock
7. Run the locked parameters on train, validation and OOS (OOS touched once)
8. Walk-forward, Monte Carlo, prop firm sizing and the arithmetic ceiling

Usage:
    uv run python -m src.main
"""

import os
import time

import pandas as pd

from src.analysis.metrics import compute_all_metrics, compute_scalping_metrics
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.parameter_selection import evaluate_params, select_parameters
from src.analysis.prop_firm import (
    PropFirmRules,
    arithmetic_ceiling_pct,
    evaluate_sizes,
    trades_per_day_counts,
)
from src.analysis.walk_forward import WalkForwardAnalyzer, WalkForwardResults
from src.backtester.block_split import (
    Block,
    blocks_for,
    build_blocks,
    print_block_audit,
    simulate_blocks,
)
from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine, BacktestResult
from src.config import load_config
from src.data.fetcher import fetch_data
from src.data.preprocessor import preprocess
from src.reports.generator import generate_full_report
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.simple_strategy import SimpleStrategy
from src.strategies.volume_profile_strategy import VolumeProfileStrategy

SPLITS = ("train", "validation", "oos")

# The figure this project was previously asked to justify.
CLAIMED_ANNUAL_RETURN_PCT = 11_376.0


def save_trade_log(result: BacktestResult, output_path: str, point_value: float) -> None:
    """Save trade log as CSV file.

    Args:
        result: BacktestResult containing trades.
        output_path: Path to save the CSV file.
        point_value: Dollar value per point for P&L conversion.
    """
    columns = [
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "pnl_gross_pts", "pnl_net_pts", "pnl_gross_dollars", "pnl_net_dollars",
        "cost_pts", "cost_dollars", "exit_reason",
    ]

    if not result.trades:
        df = pd.DataFrame(columns=columns)
    else:
        records = []
        for t in result.trades:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": "Long" if t.direction == 1 else "Short",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "pnl_gross_pts": round(t.pnl_gross, 2),
                "pnl_net_pts": round(t.pnl_net, 2),
                "pnl_gross_dollars": round(t.pnl_gross * point_value, 2),
                "pnl_net_dollars": round(t.pnl_net * point_value, 2),
                "cost_pts": round(t.cost, 2),
                "cost_dollars": round(t.cost * point_value, 2),
                "exit_reason": t.exit_reason,
            })
        df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def print_metrics_summary(
    strategy_name: str,
    train_metrics: dict,
    val_metrics: dict,
    oos_metrics: dict,
    point_value: float,
    mc_results=None,
) -> None:
    """Print train / validation / OOS metrics side by side.

    Args:
        strategy_name: Name of the strategy.
        train_metrics: Metrics from the train blocks.
        val_metrics: Metrics from the validation blocks.
        oos_metrics: Metrics from the OOS blocks.
        point_value: Dollar value per point.
        mc_results: Optional Monte Carlo results to append.
    """
    print(f"\n{'=' * 68}")
    print(f"  {strategy_name}: TRAIN vs VALIDATION vs OOS (locked parameters)")
    print(f"{'=' * 68}")

    print(f"\n  {'Metric':<25} {'Train':>12} {'Validation':>12} {'True OOS':>12}")
    print(f"  {'-' * 63}")

    metrics_to_show = [
        ("total_trades", "Total Trades", "{:.0f}"),
        ("total_return", "Total Return ($)", "${:.2f}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
        ("max_drawdown", "Max Drawdown (pts)", "{:.2f}"),
        ("profit_factor", "Profit Factor", "{:.3f}"),
        ("win_rate", "Win Rate", "{:.1%}"),
        ("avg_trade", "Avg Trade (pts)", "{:.2f}"),
        ("expected_value", "EV Per Trade (pts)", "{:.2f}"),
        ("calmar_ratio", "Calmar Ratio", "{:.3f}"),
        ("avg_trades_per_day", "Avg Trades/Day", "{:.2f}"),
        ("avg_hold_time_minutes", "Avg Hold (min)", "{:.1f}"),
        ("max_consecutive_winners", "Max Consec Winners", "{:.0f}"),
        ("max_consecutive_losers", "Max Consec Losers", "{:.0f}"),
        ("ev_per_trade_dollars", "EV/Trade ($)", "${:.2f}"),
    ]

    def format_val(key, value, fmt):
        if key == "total_return":
            return f"${value * point_value:.2f}"
        if key == "win_rate":
            return f"{value:.1%}"
        if key in ("total_trades", "max_consecutive_winners", "max_consecutive_losers"):
            return f"{int(value)}"
        if key == "ev_per_trade_dollars":
            return f"${value:.2f}"
        return fmt.format(value)

    for key, label, fmt in metrics_to_show:
        train_str = format_val(key, train_metrics.get(key, 0), fmt)
        val_str = format_val(key, val_metrics.get(key, 0), fmt)
        oos_str = format_val(key, oos_metrics.get(key, 0), fmt)
        print(f"  {label:<25} {train_str:>12} {val_str:>12} {oos_str:>12}")

    if mc_results is not None:
        print(f"\n  Monte Carlo on OOS trades ({mc_results.n_simulations:,} sims):")
        print(f"    Median Final Equity:   ${mc_results.median_final_equity:,.2f}")
        print(f"    95% CI:                [${mc_results.confidence_interval_lower:,.2f}, "
              f"${mc_results.confidence_interval_upper:,.2f}]")
        print(f"    Worst 5% Drawdown:     ${mc_results.worst_5pct_drawdown:,.2f}")
        print(f"    Probability of Ruin:   {mc_results.probability_of_ruin:.2%}")


def split_trading_days(df: pd.DataFrame, blocks: list[Block]) -> int:
    """Count distinct trading days covered by a set of blocks.

    Args:
        df: Full session-filtered DataFrame.
        blocks: Blocks belonging to one split.

    Returns:
        Number of distinct calendar dates in those blocks.
    """
    days: set = set()
    for b in blocks:
        days |= set(pd.Series(df.index[b.start_pos : b.end_pos]).dt.date)
    return len(days)


def print_overnight_hold_diagnostic(result: BacktestResult, point_value: float) -> None:
    """Report how many trades were held past the session they were opened in.

    Overnight bars are filtered out before the strategy runs, so a position
    held past 16:00 ET skips the overnight tape entirely: its stop and target
    are only ever tested against session bars. That makes overnight gap risk
    invisible, so the share of P&L coming from multi-session holds matters.

    Args:
        result: Backtest result to inspect.
        point_value: Dollar value per point.
    """
    trades = result.trades
    if not trades:
        return

    overnight = [
        t for t in trades
        if t.entry_time is not None
        and t.exit_time is not None
        and t.entry_time.date() != t.exit_time.date()
    ]
    total_pnl = sum(t.pnl_net for t in trades)
    overnight_pnl = sum(t.pnl_net for t in overnight)

    print(f"\n  Overnight-hold exposure (CAVEAT):")
    print(f"    Trades held past the entry session: {len(overnight)} of {len(trades)} "
          f"({len(overnight) / len(trades):.1%})")
    print(f"    Their share of net P&L:             "
          f"${overnight_pnl * point_value:,.0f} of ${total_pnl * point_value:,.0f}")
    print(f"    Overnight bars are filtered out before the strategy runs, so these")
    print(f"    trades never had their stop tested against the overnight tape. A gap")
    print(f"    through the stop is not modelled, which flatters these results.")


def print_scalping_summary(
    result: BacktestResult, point_value: float, trading_days: int | None = None
) -> None:
    """Print scalping-specific summary for a result.

    Args:
        result: Backtest result to summarize.
        point_value: Dollar value per point.
        trading_days: Trading days the result's blocks actually cover.
    """
    print(f"\n{'=' * 68}")
    print("  SCALPING SUMMARY (True OOS, 5-min bars)")
    print(f"{'=' * 68}")

    scalping = compute_scalping_metrics(
        result.trades, total_trading_days=trading_days, point_value=point_value
    )

    print(f"    Avg Trades/Day:        {scalping['avg_trades_per_day']:.2f}")
    print(f"    Avg Hold Time:         {scalping['avg_hold_time_minutes']:.1f} min")
    print(f"    Max Consec Winners:    {scalping['max_consecutive_winners']}")
    print(f"    Max Consec Losers:     {scalping['max_consecutive_losers']}")
    print(f"    EV/Trade:              ${scalping['ev_per_trade_dollars']:.2f}")

    session = scalping["session_breakdown"]
    if session["am"]["trades"] > 0 or session["pm"]["trades"] > 0:
        print(f"\n    Session Breakdown:")
        print(f"      AM (pre-12:00):    {session['am']['trades']} trades, "
              f"WR: {session['am']['win_rate']:.1%}, "
              f"Avg: {session['am']['avg_pnl']:.2f} pts")
        print(f"      PM (12:00+):       {session['pm']['trades']} trades, "
              f"WR: {session['pm']['win_rate']:.1%}, "
              f"Avg: {session['pm']['avg_pnl']:.2f} pts")

    if scalping["best_hour"] is not None:
        print(f"    Best Hour:             {scalping['best_hour']:02d}:00")
        print(f"    Worst Hour:            {scalping['worst_hour']:02d}:00")


def _ensure_trade_aggregation(data_config: dict) -> None:
    """Ensure trade data has been aggregated into 5-min bars.

    Args:
        data_config: Data configuration section.
    """
    from src.config import get_project_root

    project_root = get_project_root()
    data_file = data_config.get("data_file", "data/NQ_5min_real_orderflow.parquet")
    data_path = project_root / data_file

    if data_path.exists():
        print("  Real order flow data already aggregated.")
        return

    trades_dir = project_root / data_config.get("trades_dir", "data/trades")
    trade_files = list(trades_dir.glob("NQ_trades_*.parquet")) if trades_dir.exists() else []
    if not trade_files:
        print("  No raw trade files found, will use fallback data source.")
        return

    print(f"  Found {len(trade_files)} quarterly trade files, aggregating...")
    from src.data.trade_aggregator import aggregate_all_trades
    aggregate_all_trades(
        trades_dir=str(trades_dir),
        output_file=str(data_path),
        verbose=True,
    )


def _filter_session_hours(df: pd.DataFrame, session_start: str, session_end: str) -> pd.DataFrame:
    """Filter data to trading session hours only.

    Args:
        df: Preprocessed DataFrame with DatetimeIndex.
        session_start: Session start time HH:MM.
        session_end: Session end time HH:MM.

    Returns:
        DataFrame filtered to session hours only.
    """
    if not hasattr(df.index, "hour"):
        return df

    start_parts = session_start.split(":")
    end_parts = session_end.split(":")
    start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
    end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

    bar_minutes = df.index.hour * 60 + df.index.minute
    mask = (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)

    return df.loc[mask].copy()


def build_engine(
    strategy,
    config: dict,
    max_trades_per_day: int,
    session_start: str,
    session_end: str,
    session_tz: str,
) -> BacktestEngine:
    """Create an engine with fixed-point exits and block-safe closing.

    Args:
        strategy: Strategy instance.
        config: Full config dict.
        max_trades_per_day: Daily entry cap enforced by the engine.
        session_start: Session start HH:MM.
        session_end: Session end HH:MM.
        session_tz: Session timezone.

    Returns:
        Configured BacktestEngine.
    """
    return BacktestEngine(
        strategy=strategy,
        cost_model=CostModel.from_config(config.get("costs", {})),
        initial_capital=100_000.0,
        position_size=1,
        max_trades_per_day=max_trades_per_day,
        trading_session_start=session_start,
        trading_session_end=session_end,
        session_timezone=session_tz,
        exit_management=config.get("exit_management", {}),
        close_at_end=True,
    )


def print_selection_report(
    train_scores: list,
    validation_scores: list,
    locked: dict,
    point_value: float,
    top_n: int,
) -> None:
    """Print the parameter selection trail.

    Args:
        train_scores: All train scores, best first.
        validation_scores: Finalist validation scores, best first.
        locked: The locked parameter dict.
        point_value: Dollar value per point.
        top_n: How many finalists were promoted to validation.
    """
    print(f"\n  Grid scored on TRAIN only ({len(train_scores)} combinations).")
    print(f"  Score = total net P&L in points over the train blocks.")

    profitable = [s for s in train_scores if s.score > 0]
    print(f"\n  Grid-wide TRAIN result: {len(profitable)} of {len(train_scores)} "
          f"combinations profitable")
    print(f"    best   {train_scores[0].score:>10.1f} pts "
          f"(${train_scores[0].score * point_value:>10,.0f})")
    print(f"    median {train_scores[len(train_scores) // 2].score:>10.1f} pts")
    print(f"    worst  {train_scores[-1].score:>10.1f} pts")
    if not profitable:
        print(f"    NO combination in the grid is profitable on TRAIN. Anything the")
        print(f"    validation and OOS blocks show afterwards is unsupported by train.")

    print(f"\n  Top {top_n} on TRAIN (these and only these went to validation):")
    header = (
        f"    {'stop':>5} {'target':>7} {'delta':>6} {'prox':>5} "
        f"{'trades':>7} {'net pts':>10} {'net $':>12} {'EV pts':>8} {'WR':>7}"
    )
    print(header)
    for s in train_scores[:top_n]:
        p = s.params
        print(f"    {p['stop_points']:>5} {p['target_points']:>7} "
              f"{p['delta_threshold']:>6} {p['level_proximity_points']:>5} "
              f"{s.n_trades:>7} {s.total_pnl_points:>10.1f} "
              f"{s.total_pnl_points * point_value:>12,.0f} "
              f"{s.ev_per_trade_points:>8.2f} {s.win_rate:>7.1%}")

    print(f"\n  Finalists re-scored on VALIDATION (best first):")
    print(header)
    for s in validation_scores:
        p = s.params
        print(f"    {p['stop_points']:>5} {p['target_points']:>7} "
              f"{p['delta_threshold']:>6} {p['level_proximity_points']:>5} "
              f"{s.n_trades:>7} {s.total_pnl_points:>10.1f} "
              f"{s.total_pnl_points * point_value:>12,.0f} "
              f"{s.ev_per_trade_points:>8.2f} {s.win_rate:>7.1%}")

    print(f"\n  LOCKED PARAMETERS (validation winner, OOS not yet touched):")
    for key in ("stop_points", "target_points", "delta_threshold", "level_proximity_points"):
        print(f"    {key:<26} {locked[key]}")


def print_baseline_comparison(
    df: pd.DataFrame,
    blocks: list[Block],
    config: dict,
    locked: dict,
    max_trades_per_day: int,
    session_start: str,
    session_end: str,
    session_tz: str,
    point_value: float,
    simple_oos_metrics: dict,
) -> None:
    """Compare the simple strategy against the two source baselines on OOS.

    The baselines use the same locked fixed stop and target, so only the
    signal logic differs.

    Args:
        df: Full session-filtered DataFrame.
        blocks: OOS blocks.
        config: Full config dict.
        locked: Locked stop/target parameters.
        max_trades_per_day: Daily entry cap.
        session_start: Session start HH:MM.
        session_end: Session end HH:MM.
        session_tz: Session timezone.
        point_value: Dollar value per point.
        simple_oos_metrics: OOS metrics of the primary strategy.
    """
    print(f"\n{'=' * 68}")
    print("  BASELINE COMPARISON ON THE SAME OOS BLOCKS (same fixed exits)")
    print(f"{'=' * 68}")

    baseline_exits = {
        "stop_loss_mode": "fixed",
        "stop_loss_fixed_points": locked["stop_points"],
        "take_profit_fixed_points": locked["target_points"],
        "extended_take_profit_fixed_points": locked["target_points"],
        "feature_zscore_threshold": 1e9,  # never use an extended target
    }

    rows = [(
        "SimpleStrategy (primary)",
        int(simple_oos_metrics.get("total_trades", 0)),
        simple_oos_metrics.get("win_rate", 0.0),
        simple_oos_metrics.get("ev_per_trade_dollars", 0.0),
        simple_oos_metrics.get("total_return", 0.0) * point_value,
    )]

    for name, cls, section in (
        ("OrderFlowStrategy (baseline)", OrderFlowStrategy, "order_flow_strategy"),
        ("VolumeProfileStrategy (baseline)", VolumeProfileStrategy, "volume_profile_strategy"),
    ):
        params = dict(config.get(section, {}))
        params.update(baseline_exits)
        strategy = cls(params=params)
        engine = build_engine(
            strategy, config, max_trades_per_day, session_start, session_end, session_tz
        )
        signals = strategy.generate_signals(df)["signal"]
        result = simulate_blocks(engine, df, signals, blocks)
        metrics = compute_all_metrics(result, point_value)
        rows.append((
            name,
            len(result.trades),
            metrics.get("win_rate", 0.0),
            metrics.get("ev_per_trade_dollars", 0.0),
            metrics.get("total_return", 0.0) * point_value,
        ))

    print(f"\n  {'Strategy':<34} {'Trades':>7} {'Win Rate':>9} {'EV/Trade':>10} {'Total $':>12}")
    print(f"  {'-' * 74}")
    for name, trades, win_rate, ev, total in rows:
        print(f"  {name:<34} {trades:>7} {win_rate:>9.1%} {ev:>10.2f} {total:>12,.0f}")


def print_prop_firm_monte_carlo(
    oos_result: BacktestResult,
    config: dict,
    locked: dict,
    oos_trading_days: int,
    point_value: float,
) -> dict:
    """Print the FundedNext 50K sizing analysis and the arithmetic ceiling.

    Args:
        oos_result: True OOS backtest result (source of every trade P&L).
        config: Full config dict.
        locked: Locked strategy parameters (stop_points is used for sizing).
        oos_trading_days: Number of trading days observed in the OOS blocks.
        point_value: Dollar value per point of the backtested contract.

    Returns:
        Dict with the best sizing, its median return, and the ceiling.
    """
    print(f"\n{'=' * 68}")
    print("  PROP FIRM SIZING (FundedNext 50K) - TRUE OOS TRADES ONLY")
    print(f"{'=' * 68}")

    rules = PropFirmRules.from_config(config)
    prop_cfg = config.get("prop_firm", {})
    n_sims = int(prop_cfg.get("simulations", 10_000))
    seed = int(prop_cfg.get("seed", 42))

    trades = oos_result.trades
    stop_points = float(locked["stop_points"])
    day_counts = trades_per_day_counts(trades, oos_trading_days)
    observed_trades_per_day = float(day_counts.mean()) if len(day_counts) else 0.0
    ev_points = (
        sum(t.pnl_net for t in trades) / len(trades) if trades else 0.0
    )

    print(f"\n  Rules:      account ${rules.account_size:,.0f} | "
          f"target ${rules.profit_target:,.0f} | "
          f"daily loss limit ${rules.daily_loss_limit:,.0f} | "
          f"equity floor ${rules.equity_floor:,.0f}")
    print(f"  Contracts:  up to {rules.max_micro_contracts} Micro "
          f"(${rules.micro_point_value:.0f}/pt each) or "
          f"{rules.max_mini_contracts} Mini (${rules.mini_point_value:.0f}/pt each)")
    print(f"  Inputs:     {len(trades)} OOS trades over {oos_trading_days} OOS trading days")
    print(f"              locked stop = {stop_points:.0f} pts, "
          f"EV/trade = {ev_points:.2f} pts (${ev_points * point_value:.2f} on 1 Mini)")
    print(f"              observed entries/day = {observed_trades_per_day:.2f}")

    results = evaluate_sizes(
        trades=trades,
        trading_days=oos_trading_days,
        stop_points=stop_points,
        rules=rules,
        n_sims=n_sims,
        seed=seed,
    )

    print(f"\n  Step 1: eliminate sizes where ONE stop-out alone exceeds the "
          f"${rules.daily_loss_limit:,.0f} daily loss limit")
    print(f"    One stop-out = {stop_points:.0f} pts x $/pt x contracts.")
    for contract_type in ("Micro", "Mini"):
        eliminated = [r for r in results if r.contract_type == contract_type and r.eliminated]
        survivors = [r for r in results if r.contract_type == contract_type and not r.eliminated]
        if eliminated:
            lo, hi = eliminated[0], eliminated[-1]
            print(f"    {contract_type}: sizes {lo.contracts}-{hi.contracts} ELIMINATED "
                  f"(one stop-out ${lo.stop_out_dollars:,.0f}-${hi.stop_out_dollars:,.0f} "
                  f"> ${rules.daily_loss_limit:,.0f})")
        else:
            print(f"    {contract_type}: no size eliminated")
        if survivors:
            print(f"    {contract_type}: sizes {survivors[0].contracts}-{survivors[-1].contracts} "
                  f"survive (one stop-out up to ${survivors[-1].stop_out_dollars:,.0f})")

    survivors = [r for r in results if not r.eliminated]
    print(f"\n  Step 2: bar-by-bar Monte Carlo for the {len(survivors)} surviving sizes "
          f"({n_sims:,} simulated years x {rules.trading_days_per_year} days)")
    print(f"    Trades are resampled from the OOS trade list and replayed one at a")
    print(f"    time. The daily loss limit and the ${rules.equity_floor:,.0f} equity floor")
    print(f"    are hard fails checked inside the loop: on a breach the account is")
    print(f"    dead and keeps the equity it had at that instant. Nothing is capped.")

    print(f"\n  {'Size':<10} {'1 stop':>9} {'P(survive)':>11} {'P(pass $3k)':>12} "
          f"{'median':>9} {'95th pct':>10} {'max obs':>10}")
    print(f"  {'-' * 74}")
    for r in survivors:
        print(f"  {r.label:<10} {r.stop_out_dollars:>9,.0f} {r.p_survive:>11.1%} "
              f"{r.p_reach_target:>12.1%} {r.median_return_pct:>8.1f}% "
              f"{r.p95_return_pct:>9.1f}% {r.max_return_pct:>9.1f}%")

    # Best sizing: highest median annual return among sizes that survive the
    # year at least 90% of the time.
    robust = [r for r in survivors if r.p_survive >= 0.90]
    pool = robust if robust else survivors
    best = max(pool, key=lambda r: r.median_return_pct) if pool else None

    if best is not None:
        note = "P(survive) >= 90%" if robust else "no size survives 90% of years"
        print(f"\n  Best sizing ({note}): {best.label}")
        print(f"    P(survive the year):   {best.p_survive:.1%}")
        print(f"    P(reach ${rules.profit_target:,.0f} target): {best.p_reach_target:.1%}")
        print(f"    Median annual return:  {best.median_return_pct:.1f}% "
              f"(${best.median_return_pct / 100 * rules.account_size:,.0f})")
        print(f"    95th pct annual:       {best.p95_return_pct:.1f}%")
        print(f"    Max observed annual:   {best.max_return_pct:.1f}%")

    ceiling = arithmetic_ceiling_pct(ev_points, observed_trades_per_day, rules)
    max_dpp = rules.micro_point_value * rules.max_micro_contracts
    annual_dollars = ceiling / 100.0 * rules.account_size

    print(f"\n  ARITHMETIC CEILING IMPOSED BY THE {rules.max_micro_contracts}-MICRO CAP")
    print(f"    Assumptions: {rules.max_micro_contracts} Micro on every trade "
          f"(${max_dpp:.0f}/pt), {observed_trades_per_day:.2f} trades/day,")
    print(f"    {rules.trading_days_per_year} trading days, the OOS EV of "
          f"{ev_points:.2f} pts earned on every trade, zero losing")
    print(f"    days, zero daily-limit breaches, no compounding (the contract cap")
    print(f"    does not grow with equity, so accumulation is linear).")
    print(f"    Ceiling = {rules.trading_days_per_year} x {observed_trades_per_day:.2f} x "
          f"{ev_points:.2f} pts x ${max_dpp:.0f}/pt = ${annual_dollars:,.0f}")
    print(f"    Ceiling as annual return: {ceiling:.1f}%")
    micro_survivors = [r for r in survivors if r.contract_type == "Micro"]
    if any(r.eliminated and r.contract_type == "Micro" for r in results):
        largest = micro_survivors[-1].label if micro_survivors else "none"
        print(f"    NOTE: {rules.max_micro_contracts} Micro is itself ELIMINATED by the "
              f"daily loss limit, so")
        print(f"    this ceiling is not even reachable. The largest legal size is "
              f"{largest}.")

    gap = CLAIMED_ANNUAL_RETURN_PCT / ceiling if ceiling > 0 else float("inf")
    print(f"\n    Claimed figure:  {CLAIMED_ANNUAL_RETURN_PCT:,.0f}%")
    print(f"    Hard ceiling:    {ceiling:.1f}%")
    if ceiling > 0:
        print(f"    GAP: the claim is {gap:,.0f}x the arithmetic ceiling. It is not")
        print(f"    reachable under FN 50K rules with this strategy - not because the")
        print(f"    strategy is weak, but because the 40-Micro contract cap bounds the")
        print(f"    dollars per point and the account cannot compound past it.")
    else:
        print(f"    GAP: the ceiling is not positive, so the claim is unreachable.")

    return {
        "best": best,
        "ceiling_pct": ceiling,
        "gap": gap,
        "observed_trades_per_day": observed_trades_per_day,
        "ev_points": ev_points,
    }


def print_final_summary(
    locked: dict,
    train_metrics: dict,
    val_metrics: dict,
    oos_metrics: dict,
    wf_results: WalkForwardResults | None,
    mc_results,
    prop: dict,
    point_value: float,
    grid_had_profitable_train_combo: bool,
) -> None:
    """Print the final summary of everything that was measured.

    Args:
        locked: Locked parameters.
        train_metrics: Train metrics for the locked parameters.
        val_metrics: Validation metrics for the locked parameters.
        oos_metrics: True OOS metrics.
        wf_results: Walk-forward results (or None).
        mc_results: Monte Carlo results (or None).
        prop: Output of print_prop_firm_monte_carlo.
        point_value: Dollar value per point.
        grid_had_profitable_train_combo: Whether any grid combo made money on train.
    """
    print(f"\n{'=' * 68}")
    print("  FINAL SUMMARY - SimpleStrategy, interleaved-block split")
    print(f"{'=' * 68}")

    print("\n  LOCKED PARAMETERS")
    print(f"    stop_points              {locked['stop_points']}")
    print(f"    target_points            {locked['target_points']}")
    print(f"    delta_threshold          {locked['delta_threshold']}")
    print(f"    level_proximity_points   {locked['level_proximity_points']}")
    print(f"    profile_lookback         {locked['profile_lookback']}")
    print(f"    max_trades_per_day       {locked['max_trades_per_day']}")
    print(f"    session                  {locked['trading_session_start']}-"
          f"{locked['trading_session_end']} ET")

    print("\n  TRUE OOS (touched once, after the parameters were locked)")
    print(f"    Trades                   {int(oos_metrics.get('total_trades', 0))}")
    print(f"    Win rate                 {oos_metrics.get('win_rate', 0.0):.1%}")
    print(f"    EV/trade                 {oos_metrics.get('expected_value', 0.0):.2f} pts "
          f"(${oos_metrics.get('ev_per_trade_dollars', 0.0):.2f} on 1 Mini)")
    print(f"    Total net                ${oos_metrics.get('total_return', 0.0) * point_value:,.2f}")
    print(f"    Profit factor            {oos_metrics.get('profit_factor', 0.0):.3f}")
    print(f"    Sharpe                   {oos_metrics.get('sharpe_ratio', 0.0):.3f}")
    print(f"    Max drawdown             {oos_metrics.get('max_drawdown', 0.0):.2f} pts")
    print(f"    Max consecutive losers   {int(oos_metrics.get('max_consecutive_losers', 0))}")
    print(f"    Avg trades/day           {oos_metrics.get('avg_trades_per_day', 0.0):.2f}")

    print("\n  WALK-FORWARD")
    if wf_results and wf_results.total_oos_windows > 0:
        print(f"    Windows                  {wf_results.total_oos_windows}")
        print(f"    Profitable windows       {wf_results.profitable_windows}")
        print(f"    Consistency              {wf_results.consistency_ratio:.1%}")
        print(f"    Efficiency (OOS/train)   {wf_results.walk_forward_efficiency:.3f}")
        print(f"    Degradation (Sharpe)     {wf_results.degradation_metric:.3f}")
    else:
        print("    Not run")

    print("\n  MONTE CARLO (OOS trades, 1 Mini)")
    if mc_results is not None:
        print(f"    Median final equity      ${mc_results.median_final_equity:,.2f}")
        print(f"    Probability of ruin      {mc_results.probability_of_ruin:.2%}")
    else:
        print("    Not run")

    print("\n  PROP FIRM (FundedNext 50K)")
    best = prop.get("best")
    if best is not None:
        print(f"    Best legal sizing        {best.label} "
              f"(one stop-out ${best.stop_out_dollars:,.0f})")
        print(f"    P(survive the year)      {best.p_survive:.1%}")
        print(f"    Realistic median return  {best.median_return_pct:.1f}% / year")
        print(f"    95th percentile          {best.p95_return_pct:.1f}% / year")
        print(f"    Max observed             {best.max_return_pct:.1f}% / year")
    else:
        print("    No size survives the daily loss limit")
    print(f"    Arithmetic ceiling       {prop['ceiling_pct']:.1f}% / year "
          f"(40 Micro, no losing days)")
    print(f"    Claimed {CLAIMED_ANNUAL_RETURN_PCT:,.0f}%            "
          f"{prop['gap']:,.0f}x above the ceiling - unreachable")

    print("\n  VERDICT")
    checks = [
        ("Train profitable", train_metrics.get("expected_value", 0.0) > 0,
         f"EV {train_metrics.get('expected_value', 0.0):+.2f} pts/trade"),
        ("Validation profitable", val_metrics.get("expected_value", 0.0) > 0,
         f"EV {val_metrics.get('expected_value', 0.0):+.2f} pts/trade"),
        ("True OOS profitable", oos_metrics.get("expected_value", 0.0) > 0,
         f"EV {oos_metrics.get('expected_value', 0.0):+.2f} pts/trade"),
        ("Walk-forward > 50%", bool(wf_results and wf_results.consistency_ratio > 0.50),
         f"{wf_results.consistency_ratio:.0%} of windows" if wf_results else "not run"),
        ("MC ruin < 20%", bool(mc_results and mc_results.probability_of_ruin < 0.20),
         f"{mc_results.probability_of_ruin:.1%}" if mc_results else "not run"),
        ("Some grid combo works on train", grid_had_profitable_train_combo,
         "at least one combination profitable on train"),
    ]
    for label, passed, detail in checks:
        print(f"    {'PASS' if passed else 'FAIL'}  {label:<32} {detail}")

    passes = sum(1 for _, p, _ in checks if p)
    print(f"\n    {passes}/{len(checks)} checks pass.")
    if not grid_had_profitable_train_combo:
        print("    The train blocks reject this strategy: no parameter combination in")
        print("    the grid is profitable there. Positive validation and OOS numbers")
        print("    are therefore NOT evidence of an edge - the split assigns fixed")
        print("    calendar months to each split, so they are as likely to be")
        print("    seasonality as skill. Do not trade this.")
    elif not (train_metrics.get("expected_value", 0.0) > 0
              and oos_metrics.get("expected_value", 0.0) > 0):
        print("    Train and OOS do not agree. Treat the edge as unproven.")

    print(f"\n  Every number above comes from the bar-by-bar simulation or from a")
    print(f"  Monte Carlo that replays those simulated trades one at a time.")
    print(f"{'=' * 68}")


def main() -> None:
    """Run the complete pipeline on 5-min NQ data with real order flow."""
    pipeline_start = time.time()

    print("=" * 68)
    print("  NAS100 Backtesting Framework - SimpleStrategy Pipeline")
    print("=" * 68)

    # Step 1: Configuration
    step_start = time.time()
    print("\n[1/8] Loading configuration...")
    config = load_config()
    data_config = config.get("data", {})
    simple_config = dict(config.get("simple_strategy", {}))
    costs_config = config.get("costs", {})
    wf_config = config.get("walk_forward", {})
    selection_config = config.get("parameter_selection", {})
    point_value = costs_config.get("point_value", 20.0)

    max_trades_per_day = int(simple_config.get("max_trades_per_day", 2))
    session_start = simple_config.get("trading_session_start", "09:30")
    session_end = simple_config.get("trading_session_end", "16:00")
    session_tz = simple_config.get("session_timezone", "US/Eastern")

    print(f"  Data source: {data_config.get('source', 'real_orderflow')}")
    print(f"  Data file: {data_config.get('data_file')}")
    print(f"  Primary strategy: SimpleStrategy (heavy-volume level + delta)")
    print(f"  Exits: fixed stop / fixed target, no partials, no trailing")
    print(f"  Split: interleaved {data_config.get('block_months', 1)}-month blocks, "
          f"assignment {data_config.get('block_assignment')}")
    print(f"  Max trades/day: {max_trades_per_day}, session {session_start}-{session_end} ET")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 2: Trade aggregation
    step_start = time.time()
    print("\n[2/8] Checking real trade aggregation...")
    _ensure_trade_aggregation(data_config)
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 3: Load data
    step_start = time.time()
    print("\n[3/8] Loading 5-minute NQ futures data with real order flow...")
    df_raw = fetch_data(config)
    has_real_of = all(c in df_raw.columns for c in ["bid_volume", "ask_volume", "delta"])
    print(f"  Data range: {df_raw.index[0]} to {df_raw.index[-1]}")
    print(f"  Total bars: {len(df_raw):,}")
    print(f"  Real order flow: {'YES' if has_real_of else 'NO (using OHLCV proxies)'}")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 4: Preprocess
    step_start = time.time()
    print("\n[4/8] Preprocessing data...")
    df = preprocess(df_raw)
    print(f"  Preprocessed: {len(df):,} bars with {len(df.columns)} columns")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 5: Session filter
    step_start = time.time()
    print(f"\n[5/8] Filtering to session hours ({session_start}-{session_end} ET)...")
    full_bar_count = len(df)
    df = _filter_session_hours(df, session_start, session_end)
    reduction = (1 - len(df) / full_bar_count) * 100 if full_bar_count else 0.0
    print(f"  Session filter: {full_bar_count:,} -> {len(df):,} bars ({reduction:.0f}% reduction)")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 6: Interleaved-block split
    step_start = time.time()
    print("\n[6/8] Building interleaved-block split...")
    blocks = build_blocks(
        df,
        block_months=int(data_config.get("block_months", 1)),
        assignment=tuple(data_config.get("block_assignment", ("train", "train", "validation", "oos"))),
    )
    print_block_audit(blocks, SPLITS)
    train_blocks = blocks_for(blocks, "train")
    val_blocks = blocks_for(blocks, "validation")
    oos_blocks = blocks_for(blocks, "oos")
    print(f"\n  Step: {time.time() - step_start:.1f}s")

    # Step 7: Parameter selection then the locked run
    step_start = time.time()
    print("\n[7/8] Selecting parameters on TRAIN, choosing on VALIDATION...")
    strategy = SimpleStrategy(params={
        "profile_lookback": simple_config.get("profile_lookback", 78),
        "level_proximity_points": simple_config.get("level_proximity_points", 10),
        "delta_threshold": simple_config.get("delta_threshold", 1.0),
        "stop_points": simple_config.get("stop_points", 20),
        "target_points": simple_config.get("target_points", 30),
    })
    engine = build_engine(
        strategy, config, max_trades_per_day, session_start, session_end, session_tz
    )

    grid = {
        "stop_points": selection_config.get("stop_points", [15, 20, 25, 30]),
        "target_points": selection_config.get("target_points", [20, 30, 40, 45, 60]),
        "delta_threshold": selection_config.get("delta_threshold", [0.5, 1.0, 1.5]),
        "level_proximity_points": selection_config.get("level_proximity_points", [5, 10]),
    }
    top_n = int(selection_config.get("top_n_for_validation", 5))

    locked, train_scores, validation_scores = select_parameters(
        engine=engine,
        df=df,
        train_blocks=train_blocks,
        validation_blocks=val_blocks,
        grid=grid,
        min_trades=int(selection_config.get("min_trades", 30)),
        top_n=top_n,
    )
    print_selection_report(train_scores, validation_scores, locked, point_value, top_n)

    signal_cache: dict = {}
    train_score = evaluate_params(engine, df, train_blocks, locked, signal_cache)
    val_score = evaluate_params(engine, df, val_blocks, locked, signal_cache)
    oos_score = evaluate_params(engine, df, oos_blocks, locked, signal_cache)
    all_score = evaluate_params(engine, df, blocks, locked, signal_cache)

    train_result = train_score.result
    val_result = val_score.result
    oos_result = oos_score.result

    assert len(all_score.result.trades) == (
        len(train_result.trades) + len(val_result.trades) + len(oos_result.trades)
    ), "Split trades do not partition the full-history trades"

    train_metrics = compute_all_metrics(train_result, point_value)
    val_metrics = compute_all_metrics(val_result, point_value)
    oos_metrics = compute_all_metrics(oos_result, point_value)

    # A split's blocks are not contiguous, so trades/day must be measured
    # against the trading days that split actually owns.
    oos_trading_days = split_trading_days(df, oos_blocks)
    for metrics, split_blocks in (
        (train_metrics, train_blocks),
        (val_metrics, val_blocks),
        (oos_metrics, oos_blocks),
    ):
        days = split_trading_days(df, split_blocks)
        metrics["avg_trades_per_day"] = (
            metrics.get("total_trades", 0) / days if days else 0.0
        )
    print(f"\n  Step: {time.time() - step_start:.1f}s")

    # Step 8: Validation layers and reporting
    step_start = time.time()
    print("\n[8/8] Walk-forward, Monte Carlo, prop firm sizing...")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    mc_results = None
    if oos_result.trades:
        mc_results = MonteCarloSimulator.from_config(
            trades=oos_result.trades, config=config, initial_capital=100_000.0
        ).run_simulation()

    print_metrics_summary(
        "SimpleStrategy", train_metrics, val_metrics, oos_metrics, point_value, mc_results
    )

    wf_results = None
    if wf_config.get("enabled", True):
        wf_strategy = SimpleStrategy(params={**strategy.params, **locked})
        wf_analyzer = WalkForwardAnalyzer.from_config(
            strategy=wf_strategy,
            config=config,
            max_trades_per_day=max_trades_per_day,
            trading_session_start=session_start,
            trading_session_end=session_end,
            session_timezone=session_tz,
            close_at_end=True,
        )
        wf_results = wf_analyzer.run(df)
        print(f"\n  Walk-forward: {wf_results.total_oos_windows} windows, "
              f"{wf_results.profitable_windows} profitable "
              f"({wf_results.consistency_ratio:.1%} consistency)")
        if wf_results.regime_warning:
            print("  WARNING: performance degrading over time (regime sensitivity)")

    print_scalping_summary(oos_result, point_value, oos_trading_days)
    print_overnight_hold_diagnostic(oos_result, point_value)

    print_baseline_comparison(
        df, oos_blocks, config, locked, max_trades_per_day,
        session_start, session_end, session_tz, point_value, oos_metrics,
    )

    locked_full = {**locked, **{
        "profile_lookback": strategy.params["profile_lookback"],
        "max_trades_per_day": max_trades_per_day,
        "trading_session_start": session_start,
        "trading_session_end": session_end,
    }}
    prop = print_prop_firm_monte_carlo(
        oos_result, config, locked, oos_trading_days, point_value
    )

    # Reports and trade logs
    report_result = BacktestResult(
        trades=all_score.result.trades,
        equity_gross=all_score.result.equity_gross,
        equity_net=all_score.result.equity_net,
        signals=all_score.result.signals,
        train_result=train_result,
        validation_result=val_result,
        oos_result=oos_result,
        best_params=locked,
        strategy_name="SimpleStrategy",
    )
    report_path = os.path.join(results_dir, "simple_strategy_report.html")
    generate_full_report(
        strategy_name="SimpleStrategy (interleaved-block split)",
        backtest_result=report_result,
        mc_results=mc_results,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        oos_metrics=oos_metrics,
        point_value=point_value,
        output_path=report_path,
        walk_forward_results=wf_results,
    )
    save_trade_log(oos_result, os.path.join(results_dir, "simple_strategy_oos_trades.csv"), point_value)
    save_trade_log(report_result, os.path.join(results_dir, "simple_strategy_all_trades.csv"), point_value)

    print_final_summary(
        locked=locked_full,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        oos_metrics=oos_metrics,
        wf_results=wf_results,
        mc_results=mc_results,
        prop=prop,
        point_value=point_value,
        grid_had_profitable_train_combo=any(s.score > 0 for s in train_scores),
    )

    print(f"\n  Reports saved to: {os.path.abspath(results_dir)}/")
    for f in sorted(os.listdir(results_dir)):
        size = os.path.getsize(os.path.join(results_dir, f))
        print(f"    - {f} ({size / 1024:.1f} KB)")
    print(f"  Step: {time.time() - step_start:.1f}s")

    total_time = time.time() - pipeline_start
    print(f"\n  Total pipeline time: {total_time:.0f}s "
          f"({int(total_time // 60)}:{int(total_time % 60):02d})")
    print("\n  Done.")


if __name__ == "__main__":
    main()
