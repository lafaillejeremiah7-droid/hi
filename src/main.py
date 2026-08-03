"""Main entry point for the NAS100 backtesting framework.

Orchestrates the full anti-overfitting pipeline on 5-minute NQ futures data:
1. Load configuration
2. Load 5-min data from Databento parquet (393K bars, 5.5 years)
3. Preprocess with intraday indicators
4. Filter to session hours (9:30-16:00 ET) to reduce from ~393K to ~109K bars
5. For each strategy:
   a. Run 3-way split analysis (train -> validate -> OOS)
   b. Run Walk-Forward analysis (Combined only)
   c. Run Monte Carlo on True OOS trades
6. Generate comprehensive reports
7. Print final verdict with scalping metrics

Usage:
    uv run python -m src.main
"""

import os
import time

import numpy as np
import pandas as pd

from src.analysis.metrics import (
    compute_all_metrics,
    compute_scalping_metrics,
    compute_session_breakdown,
)
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.walk_forward import WalkForwardAnalyzer, WalkForwardResults
from src.backtester.costs import CostModel
from src.backtester.engine import BacktestEngine, BacktestResult
from src.config import load_config
from src.data.fetcher import fetch_data
from src.data.preprocessor import preprocess
from src.reports.generator import generate_full_report
from src.strategies.combined_strategy import CombinedStrategy
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.volume_profile_strategy import VolumeProfileStrategy


def save_trade_log(result: BacktestResult, output_path: str, point_value: float) -> None:
    """Save trade log as CSV file.

    Args:
        result: BacktestResult containing trades.
        output_path: Path to save the CSV file.
        point_value: Dollar value per point for P&L conversion.
    """
    if not result.trades:
        # Save empty CSV with headers
        df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price",
            "exit_price", "pnl_gross_pts", "pnl_net_pts", "pnl_gross_dollars",
            "pnl_net_dollars", "cost_pts", "cost_dollars", "exit_reason",
        ])
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
    mc_results,
    point_value: float,
) -> None:
    """Print formatted metrics summary to console.

    Args:
        strategy_name: Name of the strategy.
        train_metrics: Metrics from training period.
        val_metrics: Metrics from validation period.
        oos_metrics: Metrics from True OOS period.
        mc_results: Monte Carlo results.
        point_value: Dollar value per point.
    """
    print(f"\n{'=' * 60}")
    print(f"  {strategy_name}")
    print(f"{'=' * 60}")

    has_oos = bool(oos_metrics)
    if has_oos:
        print(f"\n  {'Metric':<25} {'Train':>12} {'Validation':>12} {'True OOS':>12}")
        print(f"  {'-' * 61}")
    else:
        print(f"\n  {'Metric':<25} {'Train':>15} {'Validation':>15}")
        print(f"  {'-' * 55}")

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
        elif key == "win_rate":
            return f"{value:.1%}"
        elif key in ("total_trades", "max_consecutive_winners", "max_consecutive_losers"):
            return f"{int(value)}"
        elif key == "ev_per_trade_dollars":
            return f"${value:.2f}"
        else:
            return fmt.format(value)

    for key, label, fmt in metrics_to_show:
        train_val = train_metrics.get(key, 0)
        val_val = val_metrics.get(key, 0)

        train_str = format_val(key, train_val, fmt)
        val_str = format_val(key, val_val, fmt)

        if has_oos:
            oos_val = oos_metrics.get(key, 0)
            oos_str = format_val(key, oos_val, fmt)
            print(f"  {label:<25} {train_str:>12} {val_str:>12} {oos_str:>12}")
        else:
            print(f"  {label:<25} {train_str:>15} {val_str:>15}")

    print(f"\n  Monte Carlo ({mc_results.n_simulations:,} simulations):")
    print(f"    Median Final Equity:   ${mc_results.median_final_equity:,.2f}")
    print(f"    95% CI:                [${mc_results.confidence_interval_lower:,.2f}, ${mc_results.confidence_interval_upper:,.2f}]")
    print(f"    Worst 5% Drawdown:     ${mc_results.worst_5pct_drawdown:,.2f}")
    print(f"    Probability of Ruin:   {mc_results.probability_of_ruin:.2%}")


def print_scalping_summary(
    primary_result: BacktestResult,
    point_value: float,
) -> None:
    """Print scalping-specific summary.

    Args:
        primary_result: Primary backtest result (5-min).
        point_value: Dollar value per point.
    """
    print(f"\n{'=' * 60}")
    print("  SCALPING SUMMARY (5-min bars)")
    print(f"{'=' * 60}")

    # Primary timeframe scalping metrics
    primary_scalping = compute_scalping_metrics(
        primary_result.trades, point_value=point_value
    )

    print(f"\n  Full Dataset:")
    print(f"    Avg Trades/Day:        {primary_scalping['avg_trades_per_day']:.2f}")
    print(f"    Avg Hold Time:         {primary_scalping['avg_hold_time_minutes']:.1f} min")
    print(f"    Max Consec Winners:    {primary_scalping['max_consecutive_winners']}")
    print(f"    Max Consec Losers:     {primary_scalping['max_consecutive_losers']}")
    print(f"    EV/Trade:              ${primary_scalping['ev_per_trade_dollars']:.2f}")

    # Session breakdown
    session = primary_scalping["session_breakdown"]
    if session["am"]["trades"] > 0 or session["pm"]["trades"] > 0:
        print(f"\n    Session Breakdown:")
        print(f"      AM (pre-12:00):    {session['am']['trades']} trades, "
              f"WR: {session['am']['win_rate']:.1%}, "
              f"Avg: {session['am']['avg_pnl']:.2f} pts")
        print(f"      PM (12:00+):       {session['pm']['trades']} trades, "
              f"WR: {session['pm']['win_rate']:.1%}, "
              f"Avg: {session['pm']['avg_pnl']:.2f} pts")

    if primary_scalping["best_hour"] is not None:
        print(f"    Best Hour:             {primary_scalping['best_hour']:02d}:00")
        print(f"    Worst Hour:            {primary_scalping['worst_hour']:02d}:00")


def run_strategy(
    strategy_name: str,
    strategy,
    df: pd.DataFrame,
    config: dict,
    results_dir: str,
    max_trades_per_day: int | None = None,
    max_hold_bars: int | None = None,
    trading_session_start: str | None = None,
    trading_session_end: str | None = None,
    session_timezone: str = "US/Eastern",
    use_three_way_split: bool = True,
) -> BacktestResult:
    """Run a complete analysis pipeline for a single strategy.

    Args:
        strategy_name: Human-readable strategy name.
        strategy: Strategy instance (BaseStrategy subclass).
        df: Preprocessed DataFrame.
        config: Full configuration dict.
        results_dir: Directory to save outputs.
        max_trades_per_day: Max trades per day limit.
        max_hold_bars: Deprecated, ignored.
        trading_session_start: Session start time HH:MM.
        trading_session_end: Session end time HH:MM.
        session_timezone: Timezone for session filtering.
        use_three_way_split: If True, use 3-way split; else 2-way.

    Returns:
        BacktestResult from the run.
    """
    costs_config = config.get("costs", {})
    data_config = config.get("data", {})
    exit_mgmt_config = config.get("exit_management", {})
    point_value = costs_config.get("point_value", 20.0)
    train_ratio = data_config.get("train_ratio", 0.50)
    validation_ratio = data_config.get("validation_ratio", 0.25)

    # Create cost model and engine with dynamic exit management
    cost_model = CostModel.from_config(costs_config)
    engine = BacktestEngine(
        strategy=strategy,
        cost_model=cost_model,
        initial_capital=100000.0,
        position_size=1,
        max_trades_per_day=max_trades_per_day,
        trading_session_start=trading_session_start,
        trading_session_end=trading_session_end,
        session_timezone=session_timezone,
        exit_management=exit_mgmt_config,
    )

    # Run backtest with 3-way or 2-way split
    print(f"\n  Running {strategy_name}...")
    if use_three_way_split:
        result = engine.run_with_three_way_split(
            df, train_ratio=train_ratio, validation_ratio=validation_ratio
        )
    else:
        result = engine.run_with_split(df, train_ratio=train_ratio)
    print(f"    Total trades: {len(result.trades)}")

    # Compute metrics for train and validation
    train_metrics = {}
    val_metrics = {}
    oos_metrics = {}

    if result.train_result:
        train_metrics = compute_all_metrics(result.train_result, point_value)
        print(f"    Train trades: {len(result.train_result.trades)}")
    if result.validation_result:
        val_metrics = compute_all_metrics(result.validation_result, point_value)
        print(f"    Validation trades: {len(result.validation_result.trades)}")
    if result.oos_result:
        oos_metrics = compute_all_metrics(result.oos_result, point_value)
        print(f"    True OOS trades: {len(result.oos_result.trades)}")

    # Run Monte Carlo on OOS trades (or validation if no OOS)
    if result.oos_result and result.oos_result.trades:
        mc_trades = result.oos_result.trades
    elif result.validation_result and result.validation_result.trades:
        mc_trades = result.validation_result.trades
    else:
        mc_trades = result.trades

    mc_sim = MonteCarloSimulator.from_config(
        trades=mc_trades,
        config=config,
        initial_capital=100000.0,
    )
    mc_results = mc_sim.run_simulation()
    print(f"    Monte Carlo complete: {mc_results.n_simulations:,} simulations")

    # Generate file-safe name
    file_name = strategy_name.lower().replace(" ", "_")

    # Generate HTML report
    report_path = os.path.join(results_dir, f"{file_name}_report.html")
    generate_full_report(
        strategy_name=strategy_name,
        backtest_result=result,
        mc_results=mc_results,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        oos_metrics=oos_metrics,
        point_value=point_value,
        output_path=report_path,
    )
    print(f"    Report saved: {report_path}")

    # Save trade log
    trades_path = os.path.join(results_dir, f"{file_name}_trades.csv")
    save_trade_log(result, trades_path, point_value)
    print(f"    Trade log saved: {trades_path}")

    # Print console summary
    print_metrics_summary(strategy_name, train_metrics, val_metrics, oos_metrics, mc_results, point_value)

    return result


def _ensure_trade_aggregation(data_config: dict) -> None:
    """Ensure trade data has been aggregated into 5-min bars.

    If the real order flow parquet doesn't exist but raw trade files do,
    runs the aggregation pipeline.

    Args:
        data_config: Data configuration section.
    """
    from pathlib import Path
    from src.config import get_project_root

    project_root = get_project_root()
    data_file = data_config.get("data_file", "data/NQ_5min_real_orderflow.parquet")
    data_path = project_root / data_file

    if data_path.exists():
        print("  Real order flow data already aggregated.")
        return

    trades_dir = project_root / data_config.get("trades_dir", "data/trades")
    if not trades_dir.exists():
        print("  No raw trade files found, will use fallback data source.")
        return

    trade_files = list(trades_dir.glob("NQ_trades_*.parquet"))
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


def _filter_session_hours(df: pd.DataFrame, combined_config: dict) -> pd.DataFrame:
    """Filter data to trading session hours only.

    Since we only trade during session hours (9:30-16:00 ET), processing
    overnight bars is wasteful. Filtering before strategy execution reduces
    bar count by ~72% (393K -> ~109K bars).

    Args:
        df: Preprocessed DataFrame with DatetimeIndex.
        combined_config: Combined strategy config with session params.

    Returns:
        DataFrame filtered to session hours only.
    """
    session_start = combined_config.get("trading_session_start", "09:30")
    session_end = combined_config.get("trading_session_end", "16:00")

    if not hasattr(df.index, "hour"):
        return df

    # Parse session times
    start_parts = session_start.split(":")
    end_parts = session_end.split(":")
    start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
    end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

    # Vectorized filter using index time components
    bar_minutes = df.index.hour * 60 + df.index.minute
    mask = (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)

    return df.loc[mask].copy()


def main() -> None:
    """Run the complete backtesting pipeline on 5-min NQ data."""
    pipeline_start = time.time()

    print("=" * 60)
    print("  NAS100 Backtesting Framework - Real Order Flow Pipeline")
    print("=" * 60)

    # Step 1: Load configuration
    step_start = time.time()
    print("\n[1/8] Loading configuration...")
    config = load_config()
    data_config = config.get("data", {})
    combined_config = config.get("combined_strategy", {})
    costs_config = config.get("costs", {})
    wf_config = config.get("walk_forward", {})
    point_value = costs_config.get("point_value", 20.0)

    print(f"  Data source: {data_config.get('source', 'real_orderflow')}")
    print(f"  Data file: {data_config.get('data_file', 'data/NQ_5min_real_orderflow.parquet')}")
    print(f"  3-Way Split: Train={data_config.get('train_ratio', 0.54):.0%}, "
          f"Val={data_config.get('validation_ratio', 0.27):.0%}, "
          f"OOS={1 - data_config.get('train_ratio', 0.54) - data_config.get('validation_ratio', 0.27):.0%}")
    print(f"  Walk-Forward: {'Enabled' if wf_config.get('enabled', True) else 'Disabled'}")
    print(f"  Max trades/day: {combined_config.get('max_trades_per_day', 2)}")
    print(f"  Exit management: Dynamic (partial close + trailing stop)")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 2: Aggregate real trade data (if needed)
    step_start = time.time()
    print("\n[2/8] Aggregating real trade data...")
    _ensure_trade_aggregation(data_config)
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 3: Load data
    step_start = time.time()
    print("\n[3/8] Loading 5-minute NQ futures data with real order flow...")
    df_raw = fetch_data(config)
    has_real_of = all(col in df_raw.columns for col in ["bid_volume", "ask_volume", "delta"])
    print(f"  Data range: {df_raw.index[0]} to {df_raw.index[-1]}")
    print(f"  Total bars: {len(df_raw):,}")
    print(f"  Real order flow: {'YES' if has_real_of else 'NO (using OHLCV proxies)'}")
    if has_real_of:
        print(f"  Columns: {list(df_raw.columns)}")

    # Estimate trading days
    if hasattr(df_raw.index, "date"):
        n_days = len(set(df_raw.index.date))
        print(f"  Trading days: {n_days:,}")
        print(f"  Avg bars/day: {len(df_raw) / n_days:.0f}")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 4: Preprocess
    step_start = time.time()
    print("\n[4/8] Preprocessing data (indicators, S/R levels)...")
    df = preprocess(df_raw)
    print(f"  Preprocessed: {len(df):,} bars with {len(df.columns)} columns")
    if has_real_of:
        has_absorption = "absorption_signal" in df.columns
        has_imbalance = "imbalance_signal" in df.columns
        print(f"  Real OF signals: absorption={has_absorption}, imbalance={has_imbalance}")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 5: Filter to session hours (9:30-16:00 ET)
    step_start = time.time()
    print("\n[5/8] Filtering to session hours (9:30-16:00 ET)...")
    full_bar_count = len(df)
    df = _filter_session_hours(df, combined_config)
    session_bar_count = len(df)
    reduction_pct = (1 - session_bar_count / full_bar_count) * 100 if full_bar_count > 0 else 0
    print(f"  Session filter: {full_bar_count:,} -> {session_bar_count:,} bars ({reduction_pct:.0f}% reduction)")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 6: Run strategies with 3-way split
    step_start = time.time()
    print("\n[6/8] Running strategies with 3-way split...")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Scalping parameters from combined config
    max_trades_per_day = combined_config.get("max_trades_per_day", 2)
    session_start = combined_config.get("trading_session_start", "09:30")
    session_end = combined_config.get("trading_session_end", "16:00")
    session_tz = combined_config.get("session_timezone", "US/Eastern")

    # Strategy 1: Order Flow (3-way split, no walk-forward)
    of_config = config.get("order_flow_strategy", {})
    of_strategy = OrderFlowStrategy(params=of_config)
    run_strategy(
        "Order Flow", of_strategy, df, config, results_dir,
        max_trades_per_day=max_trades_per_day,
        trading_session_start=session_start,
        trading_session_end=session_end,
        session_timezone=session_tz,
        use_three_way_split=True,
    )

    # Strategy 2: Volume Profile (3-way split, no walk-forward)
    vp_config = config.get("volume_profile_strategy", {})
    vp_strategy = VolumeProfileStrategy(params=vp_config)
    run_strategy(
        "Volume Profile", vp_strategy, df, config, results_dir,
        max_trades_per_day=max_trades_per_day,
        trading_session_start=session_start,
        trading_session_end=session_end,
        session_timezone=session_tz,
        use_three_way_split=True,
    )

    # Strategy 3: Combined (full anti-overfitting pipeline)
    combined_strategy = CombinedStrategy(params=combined_config)
    primary_result = run_strategy(
        "Combined", combined_strategy, df, config, results_dir,
        max_trades_per_day=max_trades_per_day,
        trading_session_start=session_start,
        trading_session_end=session_end,
        session_timezone=session_tz,
        use_three_way_split=True,
    )
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 7: Walk-Forward Analysis (Combined strategy only)
    step_start = time.time()
    print("\n[7/8] Walk-Forward Analysis (Combined only)...")
    wf_results = None
    if wf_config.get("enabled", True):
        wf_strategy = CombinedStrategy(params=combined_config)
        wf_analyzer = WalkForwardAnalyzer.from_config(
            strategy=wf_strategy,
            config=config,
            max_trades_per_day=max_trades_per_day,
            trading_session_start=session_start,
            trading_session_end=session_end,
            session_timezone=session_tz,
        )
        wf_results = wf_analyzer.run(df)
        print(f"    Walk-Forward windows: {wf_results.total_oos_windows}")
        print(f"    Profitable windows: {wf_results.profitable_windows}")
        print(f"    Consistency ratio: {wf_results.consistency_ratio:.1%}")
        print(f"    Walk-forward efficiency: {wf_results.walk_forward_efficiency:.3f}")
        print(f"    Degradation metric: {wf_results.degradation_metric:.3f}")
        if wf_results.regime_warning:
            print("    WARNING: Performance degrading over time (regime sensitivity)")
    else:
        print("  Walk-Forward Analysis: Disabled")
    print(f"  Step: {time.time() - step_start:.1f}s")

    # Step 8: Final verdict and reports
    step_start = time.time()
    print("\n[8/8] Generating reports and final verdict...")

    # Monte Carlo on True OOS trades
    mc_results = None
    if primary_result.oos_result and primary_result.oos_result.trades:
        mc_sim = MonteCarloSimulator.from_config(
            trades=primary_result.oos_result.trades,
            config=config,
            initial_capital=100000.0,
        )
        mc_results = mc_sim.run_simulation()

    # Generate comprehensive combined report with walk-forward
    if wf_results:
        combined_report_path = os.path.join(results_dir, "combined_comprehensive_report.html")
        oos_metrics = (
            compute_all_metrics(primary_result.oos_result, point_value)
            if primary_result.oos_result
            else {}
        )
        train_metrics = (
            compute_all_metrics(primary_result.train_result, point_value)
            if primary_result.train_result
            else {}
        )
        val_metrics = (
            compute_all_metrics(primary_result.validation_result, point_value)
            if primary_result.validation_result
            else {}
        )

        if mc_results is None:
            mc_sim = MonteCarloSimulator.from_config(
                trades=primary_result.trades,
                config=config,
                initial_capital=100000.0,
            )
            mc_results = mc_sim.run_simulation()

        generate_full_report(
            strategy_name="Combined (5-Min Anti-Overfitting Pipeline)",
            backtest_result=primary_result,
            mc_results=mc_results,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            oos_metrics=oos_metrics,
            point_value=point_value,
            output_path=combined_report_path,
            walk_forward_results=wf_results,
        )
        print(f"  Comprehensive report saved: {combined_report_path}")

    # Print scalping summary
    print_scalping_summary(primary_result, point_value)

    # Print FINAL VERDICT
    print_final_verdict(primary_result, wf_results, mc_results, point_value)

    # Final file listing
    print(f"\n  Reports saved to: {os.path.abspath(results_dir)}/")
    print("  Files generated:")
    for f in sorted(os.listdir(results_dir)):
        filepath = os.path.join(results_dir, f)
        size = os.path.getsize(filepath)
        print(f"    - {f} ({size / 1024:.1f} KB)")

    # Total pipeline time
    total_time = time.time() - pipeline_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n  Total pipeline time: {total_time:.0f}s ({minutes}:{seconds:02d})")
    print("\n  Done.")


def print_final_verdict(
    primary_result: BacktestResult,
    wf_results: WalkForwardResults | None,
    mc_results,
    point_value: float,
) -> None:
    """Print the final anti-overfitting verdict.

    Checks three criteria:
    1. True OOS profitable (positive EV)
    2. Walk-forward consistency > 50%
    3. Monte Carlo probability of ruin < 20%

    All must pass for the strategy to be considered viable.

    Args:
        primary_result: Primary backtest result with OOS.
        wf_results: Walk-forward results (or None).
        mc_results: Monte Carlo results.
        point_value: Dollar value per point.
    """
    print(f"\n{'=' * 60}")
    print("  FINAL VERDICT: Combined Strategy (5-Min Real Data)")
    print(f"{'=' * 60}")

    passes = 0
    total = 3

    # 1. True OOS
    oos_ev = 0.0
    oos_pass = False
    if primary_result.oos_result and primary_result.oos_result.trades:
        oos_metrics = compute_all_metrics(primary_result.oos_result, point_value)
        oos_ev = oos_metrics.get("ev_per_trade_dollars", 0.0)
        oos_pass = oos_ev > 0
    if oos_pass:
        passes += 1
        print(f"\n  True OOS (Mar 2025 - Jul 2026):  PASS (EV: +${oos_ev:.2f}/trade)")
    else:
        print(f"\n  True OOS (Mar 2025 - Jul 2026):  FAIL (EV: ${oos_ev:.2f}/trade)")

    # 2. Walk-Forward Consistency
    wf_pass = False
    wf_consistency = 0.0
    if wf_results and wf_results.total_oos_windows > 0:
        wf_consistency = wf_results.consistency_ratio
        wf_pass = wf_consistency > 0.50
    if wf_pass:
        passes += 1
        print(f"  Walk-Forward Consistency:        PASS ({wf_consistency:.0%} of months profitable)")
    else:
        print(f"  Walk-Forward Consistency:        FAIL ({wf_consistency:.0%} of months profitable)")

    # 3. Monte Carlo Ruin Probability
    mc_pass = False
    mc_ruin = 1.0
    if mc_results is not None:
        mc_ruin = mc_results.probability_of_ruin
        mc_pass = mc_ruin < 0.20
    if mc_pass:
        passes += 1
        print(f"  Monte Carlo Ruin Probability:    PASS ({mc_ruin:.1%} < 20% threshold)")
    else:
        print(f"  Monte Carlo Ruin Probability:    FAIL ({mc_ruin:.1%} >= 20% threshold)")

    # Overall
    print(f"\n  Overall: {passes}/{total} PASS", end="")
    if passes == total:
        print(" -- Strategy deemed VIABLE across all validation layers")
    elif passes >= 2:
        print(" -- Strategy shows edge but needs refinement")
    else:
        print(" -- Strategy does NOT pass anti-overfitting validation")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
