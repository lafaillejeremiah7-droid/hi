"""Larry Williams daily-bar pipeline.

End to end, in the order the work has to happen so that nothing downstream can
contaminate anything upstream:

1. Resample the 1-minute file into daily RTH bars and cache them.
2. Build the embargoed chronological split and print all five segments.
3. Measure each Williams component standalone on TRAIN at fixed default exits.
4. Decide which components to combine, on TRAIN only.
5. Score the capped grid on TRAIN, promote the top 3 to VALIDATION, lock one.
6. Touch OOS once and report train / validation / OOS side by side.
7. Walk-forward over windows from the existing WalkForwardAnalyzer.
8. Gap-fill statistics, then the FundedNext 50K sizing analysis.

If no grid combination is profitable on TRAIN the pipeline stops after step 5
and reports that as the headline. Validation and OOS numbers are not shown in
that case, because they would be noise dressed up as evidence.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.analysis.daily_selection import (
    MAX_GRID_COMBINATIONS,
    DailyComboScore,
    build_daily_combos,
    evaluate_daily_params,
    select_daily_parameters,
)
from src.analysis.metrics import compute_all_metrics
from src.analysis.prop_firm import (
    PropFirmRules,
    arithmetic_ceiling_pct,
    evaluate_sizes,
)
from src.analysis.walk_forward import WalkForwardAnalyzer
from src.backtester.costs import CostModel
from src.backtester.daily_engine import (
    DailyBacktestEngine,
    DailySimResult,
    IntradayPaths,
    daily_pnl_units,
    gap_fill_statistics,
    replay_prop_account,
)
from src.backtester.embargo_split import (
    Segment,
    build_embargo_split,
    print_embargo_audit,
    segment_by_name,
)
from src.config import get_project_root
from src.data.daily_bars import load_daily_rth_bars, overnight_gap_points
from src.indicators.williams import tdom_bias_table
from src.strategies.williams_strategy import WilliamsStrategy

# The figure this project was previously asked to justify.
CLAIMED_ANNUAL_RETURN_PCT = 11_376.0

# Fixed, declared, never tuned: the exits used when each component is measured
# standalone, so components are compared on equal terms.
STANDALONE_EXITS = {"stop_points": 50, "target_points": 100, "max_hold_days": 3}

COMPONENT_DESCRIPTIONS = {
    "gsv": "Greatest Swing Value volatility breakout",
    "oops": "Oops! open-beyond-yesterday's-extreme reversal",
    "smash": "Smash Day reversal",
    "tdom": "Trading Day of Month seasonality",
}


@dataclass
class ComponentScore:
    """Standalone train performance of one component."""

    name: str
    n_trades: int
    total_pnl_points: float
    ev_per_trade_points: float
    win_rate: float
    profit_factor: float
    max_consecutive_losers: int


@dataclass
class WilliamsOutcome:
    """Everything the pipeline measured, for the final summary."""

    daily_days: int = 0
    date_range: tuple[Any, Any] = (None, None)
    segments: list[Segment] = field(default_factory=list)
    component_scores: list[ComponentScore] = field(default_factory=list)
    inverted_gsv: ComponentScore | None = None
    combination_scores: list[tuple[str, float, int]] = field(default_factory=list)
    chosen_components: tuple[str, ...] = ()
    chosen_tdom_filter: bool = False
    combination_reason: str = ""
    train_scores: list[DailyComboScore] = field(default_factory=list)
    validation_scores: list[DailyComboScore] = field(default_factory=list)
    locked: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    oos_metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    gap_stats: dict[str, Any] = field(default_factory=dict)
    prop: dict[str, Any] = field(default_factory=dict)
    cot: dict[str, Any] = field(default_factory=dict)
    train_profitable_combos: int = 0
    total_train_evaluations: int = 0
    stopped_on_train_failure: bool = False


# --------------------------------------------------------------------- helpers


def _score_from_result(name: str, result: DailySimResult) -> ComponentScore:
    """Summarize a daily simulation as a ComponentScore.

    Args:
        name: Label for the component or combination.
        result: Simulation result.

    Returns:
        ComponentScore.
    """
    trades = result.trades
    n = len(trades)
    wins = [t.pnl_net for t in trades if t.pnl_net > 0]
    losses = [t.pnl_net for t in trades if t.pnl_net < 0]
    gross_loss = abs(sum(losses))

    streak = 0
    worst_streak = 0
    for trade in trades:
        if trade.pnl_net < 0:
            streak += 1
            worst_streak = max(worst_streak, streak)
        elif trade.pnl_net > 0:
            streak = 0

    total = sum(t.pnl_net for t in trades)
    return ComponentScore(
        name=name,
        n_trades=n,
        total_pnl_points=total,
        ev_per_trade_points=total / n if n else 0.0,
        win_rate=len(wins) / n if n else 0.0,
        profit_factor=(sum(wins) / gross_loss) if gross_loss else (float("inf") if wins else 0.0),
        max_consecutive_losers=worst_streak,
    )


def _run_once(
    engine: DailyBacktestEngine,
    df: pd.DataFrame,
    segment: Segment,
    params: dict[str, Any],
) -> DailySimResult:
    """Apply params and simulate one segment.

    Args:
        engine: Daily engine.
        df: Full daily frame.
        segment: Segment to simulate.
        params: Parameters to apply.

    Returns:
        DailySimResult.
    """
    engine.strategy.params.update(params)
    signals = engine.strategy.generate_signals(df)
    return engine.run(df, signals, start=segment.start_pos, end=segment.end_pos)


def save_daily_trade_log(result: DailySimResult, output_path: str, point_value: float) -> None:
    """Write a daily trade log including the gap-fill columns.

    Args:
        result: Simulation result to write.
        output_path: Destination CSV path.
        point_value: Dollar value per point.
    """
    records = []
    for t in result.trades:
        records.append(
            {
                "entry_date": t.entry_time,
                "exit_date": t.exit_time,
                "direction": "Long" if t.direction == 1 else "Short",
                "component": t.entry_component,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "intended_stop": round(t.intended_stop, 2),
                "intended_target": round(t.intended_target, 2),
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "gap_fill": t.gap_fill,
                "gap_slippage_pts": round(t.gap_slippage_points, 2),
                "entry_overnight_gap_pts": round(t.entry_gap_points, 2),
                "pnl_net_pts": round(t.pnl_net, 2),
                "pnl_net_dollars": round(t.pnl_net * point_value, 2),
                "cost_pts": round(t.cost, 3),
            }
        )

    columns = [
        "entry_date", "exit_date", "direction", "component", "entry_price",
        "exit_price", "intended_stop", "intended_target", "hold_days",
        "exit_reason", "gap_fill", "gap_slippage_pts", "entry_overnight_gap_pts",
        "pnl_net_pts", "pnl_net_dollars", "cost_pts",
    ]
    frame = pd.DataFrame(records, columns=columns)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame.to_csv(output_path, index=False)


# -------------------------------------------------------------------- printing


def print_component_table(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print standalone TRAIN performance of each Williams component."""
    print(f"\n{'=' * 76}")
    print("  STEP 3: EACH WILLIAMS COMPONENT STANDALONE, TRAIN ONLY")
    print(f"{'=' * 76}")
    print(f"\n  Identical fixed exits for all four so they are comparable: "
          f"{STANDALONE_EXITS['stop_points']}-pt stop,")
    print(f"  {STANDALONE_EXITS['target_points']}-pt target, "
          f"{STANDALONE_EXITS['max_hold_days']}-day maximum hold. These exits are declared, "
          f"not tuned.")

    header = (f"\n  {'Component':<52} {'Trades':>7} {'Net pts':>10} {'Net $':>12} "
              f"{'EV pts':>8} {'Win%':>7} {'PF':>6} {'MaxCL':>6}")
    print(header)
    print(f"  {'-' * 112}")
    for score in outcome.component_scores:
        label = f"{score.name}: {COMPONENT_DESCRIPTIONS[score.name]}"
        print(f"  {label:<52} {score.n_trades:>7} {score.total_pnl_points:>10.1f} "
              f"{score.total_pnl_points * point_value:>12,.0f} "
              f"{score.ev_per_trade_points:>8.2f} {score.win_rate:>6.1%} "
              f"{score.profit_factor:>6.2f} {score.max_consecutive_losers:>6}")

    if outcome.inverted_gsv is not None:
        inv = outcome.inverted_gsv
        print(f"\n  Diagnostic, not part of any grid or selection: the same GSV breakout")
        print(f"  with the swings taken from the OPPOSITE close days (the formulation")
        print(f"  several published versions of GSV use):")
        print(f"  {'gsv (inverted swing definition)':<52} {inv.n_trades:>7} "
              f"{inv.total_pnl_points:>10.1f} {inv.total_pnl_points * point_value:>12,.0f} "
              f"{inv.ev_per_trade_points:>8.2f} {inv.win_rate:>6.1%} "
              f"{inv.profit_factor:>6.2f} {inv.max_consecutive_losers:>6}")


def print_combination_choice(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print how the component combination was decided on TRAIN."""
    print(f"\n{'=' * 76}")
    print("  STEP 4: WHICH COMPONENTS GET COMBINED (TRAIN ONLY)")
    print(f"{'=' * 76}")
    print("\n  Rule applied: a component is a candidate only if it is profitable")
    print("  standalone on train. TDOM is never an entry component - Williams used")
    print("  seasonality as a bias, so it can only enter as a filter. Candidates are")
    print("  then combined cumulatively, best standalone first, and each combination")
    print("  is measured on train at the same fixed exits.")

    print(f"\n  {'Combination':<44} {'Trades':>7} {'Net pts':>10} {'Net $':>12}")
    print(f"  {'-' * 75}")
    for label, points, n_trades in outcome.combination_scores:
        print(f"  {label:<44} {n_trades:>7} {points:>10.1f} {points * point_value:>12,.0f}")

    print(f"\n  CHOSEN: {', '.join(outcome.chosen_components)}"
          f"{' + TDOM filter' if outcome.chosen_tdom_filter else ''}")
    print(f"  WHY:    {outcome.combination_reason}")


def print_selection_report(
    outcome: WilliamsOutcome, grid: dict[str, list], point_value: float, top_n: int
) -> None:
    """Print the grid, the train result, the finalists and the locked parameters."""
    train_scores = outcome.train_scores

    print(f"\n{'=' * 76}")
    print("  STEP 5: PARAMETER SELECTION (TRAIN SCORES, VALIDATION PICKS)")
    print(f"{'=' * 76}")

    print(f"\n  Grid ({len(train_scores)} combinations, hard cap {MAX_GRID_COMBINATIONS}):")
    for name, values in grid.items():
        print(f"    {name:<20} {values}")
    print(f"    target_points        fixed at 2.0 x stop_points (declared, not tuned)")
    print(f"    trailing exit        supported by the engine, disabled here (not tuned)")

    profitable = [s for s in train_scores if s.score > 0]
    print(f"\n  TRAIN result across the whole grid: {len(profitable)} of "
          f"{len(train_scores)} combinations profitable")
    print(f"    best   {train_scores[0].score:>10.1f} pts "
          f"(${train_scores[0].score * point_value:>10,.0f})")
    print(f"    median {train_scores[len(train_scores) // 2].score:>10.1f} pts")
    print(f"    worst  {train_scores[-1].score:>10.1f} pts "
          f"(${train_scores[-1].score * point_value:>10,.0f})")

    if not profitable:
        print(f"\n  NO combination in the grid is profitable on TRAIN.")
        return

    header = (f"    {'N':>3} {'mult':>5} {'stop':>5} {'hold':>5} {'trades':>7} "
              f"{'net pts':>10} {'net $':>12} {'EV pts':>8} {'Win%':>7} {'PF':>6} {'MaxCL':>6}")

    def row(score: DailyComboScore) -> str:
        p = score.params
        return (f"    {p['gsv_lookback']:>3} {p['gsv_multiplier']:>5} {p['stop_points']:>5} "
                f"{p['max_hold_days']:>5} {score.n_trades:>7} {score.total_pnl_points:>10.1f} "
                f"{score.total_pnl_points * point_value:>12,.0f} "
                f"{score.ev_per_trade_points:>8.2f} {score.win_rate:>6.1%} "
                f"{score.profit_factor:>6.2f} {score.max_consecutive_losers:>6}")

    print(f"\n  Top {top_n} on TRAIN - these and only these were run on VALIDATION:")
    print(header)
    for score in outcome.train_scores[:top_n]:
        print(row(score))

    print(f"\n  The finalists re-scored on VALIDATION (best first):")
    print(header)
    for score in outcome.validation_scores:
        print(row(score))

    print(f"\n  LOCKED PARAMETERS (validation winner; OOS still untouched):")
    for key in ("components", "gsv_lookback", "gsv_multiplier", "stop_points",
                "target_points", "max_hold_days", "tdom_filter"):
        if key in outcome.locked:
            print(f"    {key:<20} {outcome.locked[key]}")


def print_split_comparison(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print train / validation / OOS metrics side by side."""
    print(f"\n{'=' * 76}")
    print("  STEP 6: TRAIN vs VALIDATION vs TRUE OOS (locked parameters)")
    print(f"{'=' * 76}")

    rows = [
        ("total_trades", "Trades", "int"),
        ("win_rate", "Win rate", "pct"),
        ("expected_value", "EV / trade (pts)", "f2"),
        ("ev_per_trade_dollars", "EV / trade ($, 1 Mini)", "usd"),
        ("total_return", "Total net (pts)", "f1"),
        ("profit_factor", "Profit factor", "f3"),
        ("max_consecutive_losers", "Max consecutive losers", "int"),
        ("sharpe_ratio", "Sharpe (daily equity)", "f3"),
        ("max_drawdown", "Max drawdown (pts)", "f1"),
        ("avg_trade", "Avg trade (pts)", "f2"),
        ("best_trade", "Best trade (pts)", "f1"),
        ("worst_trade", "Worst trade (pts)", "f1"),
    ]

    def fmt(value, kind: str) -> str:
        if kind == "int":
            return f"{int(value)}"
        if kind == "pct":
            return f"{value:.1%}"
        if kind == "usd":
            return f"${value:,.2f}"
        if kind == "f1":
            return f"{value:.1f}"
        if kind == "f2":
            return f"{value:.2f}"
        return f"{value:.3f}"

    print(f"\n  {'Metric':<26} {'Train':>14} {'Validation':>14} {'True OOS':>14}")
    print(f"  {'-' * 70}")
    for key, label, kind in rows:
        train = fmt(outcome.train_metrics.get(key, 0), kind)
        val = fmt(outcome.validation_metrics.get(key, 0), kind)
        oos = fmt(outcome.oos_metrics.get(key, 0), kind)
        print(f"  {label:<26} {train:>14} {val:>14} {oos:>14}")

    print(f"\n  Total net in dollars on 1 Mini (${point_value:.0f}/pt):")
    for name, metrics in (("train", outcome.train_metrics),
                          ("validation", outcome.validation_metrics),
                          ("OOS", outcome.oos_metrics)):
        print(f"    {name:<12} ${metrics.get('total_return', 0.0) * point_value:>12,.0f}")


def print_walk_forward(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print the walk-forward window table."""
    wf = outcome.walk_forward
    print(f"\n{'=' * 76}")
    print("  STEP 7: WALK-FORWARD")
    print(f"{'=' * 76}")

    if not wf.get("windows"):
        print("\n  Not run: not enough daily bars for the configured windows.")
        return

    print(f"\n  Windows come from WalkForwardAnalyzer._generate_windows "
          f"({wf['train_months']}-month train,")
    print(f"  {wf['test_months']}-month test, {wf['step_months']}-month step). The "
          f"simulation itself is the daily")
    print(f"  engine, because BacktestEngine enters at a bar's close and cannot "
          f"represent a")
    print(f"  resting stop order or an overnight gap. Each window re-selects "
          f"parameters on its")
    print(f"  own train rows only, then trades the following test rows once.")

    print(f"\n  {'#':>2} {'Train end':>12} {'Test window':>25} {'Trades':>7} "
          f"{'Net pts':>10} {'Net $':>11}  Params (N/mult/stop/hold)")
    print(f"  {'-' * 96}")
    for window in wf["windows"]:
        params = window["params"]
        label = f"{window['test_start']} to {window['test_end']}"
        print(f"  {window['idx']:>2} {str(window['train_end']):>12} {label:>25} "
              f"{window['trades']:>7} {window['net_points']:>10.1f} "
              f"{window['net_points'] * point_value:>11,.0f}  "
              f"{params['gsv_lookback']}/{params['gsv_multiplier']}/"
              f"{params['stop_points']}/{params['max_hold_days']}")

    print(f"\n  Profitable windows:   {wf['profitable']} of {wf['total']} "
          f"({wf['consistency']:.1%} consistency)")
    print(f"  Combined OOS net:     {wf['total_points']:.1f} pts "
          f"(${wf['total_points'] * point_value:,.0f})")
    print(f"  Distinct locked sets: {wf['distinct_param_sets']} across "
          f"{wf['total']} windows")
    if wf["distinct_param_sets"] > 1:
        print(f"  Parameter instability is itself a result: the grid does not settle on")
        print(f"  one answer, so the 'best' parameters are period-specific.")


def print_gap_analysis(outcome: WilliamsOutcome, point_value: float, gap_series: pd.Series) -> None:
    """Print the overnight gap and gap-fill statistics."""
    stats = outcome.gap_stats
    print(f"\n{'=' * 76}")
    print("  STEP 8: OVERNIGHT GAP RISK (TRUE OOS TRADES)")
    print(f"{'=' * 76}")

    print(f"\n  Overnight gaps in the daily RTH data (open minus prior RTH close):")
    absolute = gap_series.abs().dropna()
    print(f"    median |gap|            {absolute.median():.1f} pts "
          f"(${absolute.median() * point_value:,.0f} on 1 Mini)")
    print(f"    90th pct |gap|          {absolute.quantile(0.90):.1f} pts")
    print(f"    largest |gap|           {absolute.max():.1f} pts "
          f"(${absolute.max() * point_value:,.0f} on 1 Mini)")

    overnight_share = (
        stats["overnight_trades"] / stats["total_trades"] if stats["total_trades"] else 0.0
    )
    print(f"\n  How the locked system was actually filled on OOS:")
    print(f"    Trades                  {stats['total_trades']}")
    print(f"    Held at least one night {stats['overnight_trades']} ({overnight_share:.1%})")
    print(f"    Losing trades           {stats['losing_trades']}")
    print(f"    Of those, GAP FILLS     {stats['gap_fill_losers']} "
          f"({stats['gap_fill_share_of_losers']:.1%} of losers)")
    print(f"    A gap fill is a stop that could not be honoured: the session opened")
    print(f"    beyond it, so the fill is the open.")
    print(f"    Average gap fill was    {stats['avg_gap_slippage_points']:.1f} pts worse "
          f"than the intended stop (${stats['avg_gap_slippage_points'] * point_value:,.0f})")
    print(f"    Worst gap fill was      {stats['worst_gap_slippage_points']:.1f} pts worse "
          f"(${stats['worst_gap_slippage_points'] * point_value:,.0f})")
    print(f"    Worst single trade      {stats['worst_trade_points']:.1f} pts "
          f"(${stats['worst_trade_points'] * point_value:,.0f})")

    worst_day = outcome.prop.get("worst_day_points", 0.0)
    if worst_day:
        print(f"    Worst single DAY        {worst_day:.1f} pts "
              f"(${worst_day * point_value:,.0f}) of mark-to-market equity move")


def print_prop_firm(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print the FundedNext 50K sizing analysis and the arithmetic ceiling."""
    prop = outcome.prop
    rules: PropFirmRules = prop["rules"]
    results = prop["size_results"]
    survivors = [r for r in results if not r.eliminated]

    print(f"\n{'=' * 76}")
    print("  STEP 9: FUNDEDNEXT 50K SIZING - TRUE OOS DAYS ONLY")
    print(f"{'=' * 76}")

    print(f"\n  Rules:     account ${rules.account_size:,.0f} | target "
          f"${rules.profit_target:,.0f} | daily loss limit "
          f"${rules.daily_loss_limit:,.0f} | floor ${rules.equity_floor:,.0f}")
    print(f"  Contracts: up to {rules.max_micro_contracts} Micro "
          f"(${rules.micro_point_value:.0f}/pt each) or {rules.max_mini_contracts} Mini "
          f"(${rules.mini_point_value:.0f}/pt each)")
    print(f"  Inputs:    {prop['oos_trades']} OOS trades over {prop['oos_days']} OOS "
          f"trading days, locked stop {prop['stop_points']:.0f} pts")

    print(f"\n  Step 1: eliminate any size where ONE stop-out alone exceeds the "
          f"${rules.daily_loss_limit:,.0f} daily limit")
    print(f"    One stop-out = {prop['stop_points']:.0f} pts x $/pt x contracts.")
    for contract_type in ("Micro", "Mini"):
        eliminated = [r for r in results if r.contract_type == contract_type and r.eliminated]
        kept = [r for r in results if r.contract_type == contract_type and not r.eliminated]
        if eliminated:
            print(f"    {contract_type}: sizes {eliminated[0].contracts}-"
                  f"{eliminated[-1].contracts} ELIMINATED (one stop-out "
                  f"${eliminated[0].stop_out_dollars:,.0f}-"
                  f"${eliminated[-1].stop_out_dollars:,.0f} > "
                  f"${rules.daily_loss_limit:,.0f})")
        else:
            print(f"    {contract_type}: none eliminated by this filter")
        if kept:
            print(f"    {contract_type}: sizes {kept[0].contracts}-{kept[-1].contracts} "
                  f"survive (one stop-out up to ${kept[-1].stop_out_dollars:,.0f})")
        else:
            print(f"    {contract_type}: NO size survives")

    gap_aware = prop["gap_aware_max_contracts"]
    print(f"\n  Step 1b: the same filter run against the worst OOS single-day loss "
          f"instead of the")
    print(f"    intended stop, because a stop that gets gapped through is not what the")
    print(f"    day actually costs. Worst observed day = "
          f"{prop['worst_day_points']:.1f} pts.")
    print(f"    Keeping that day inside the ${rules.daily_loss_limit:,.0f} limit allows at "
          f"most {gap_aware['micro']} Micro (that day")
    print(f"    would have cost ${gap_aware['micro_dollars']:,.0f} at that size) and "
          f"{gap_aware['mini']} Mini.")

    print(f"\n  Step 2: Monte Carlo, {prop['n_sims']:,} simulated years x "
          f"{rules.trading_days_per_year} days, for the")
    print(f"    {len(survivors)} surviving sizes. The resampled unit is one TRADING DAY's "
          f"mark-to-market")
    print(f"    equity move taken from the OOS simulation, because FundedNext's daily")
    print(f"    limit is a rule about a day's equity move, not about a trade: a position")
    print(f"    gapped through its stop breaches the limit before the trader can act.")
    print(f"    The daily limit and the ${rules.equity_floor:,.0f} floor are checked "
          f"inside the loop; on a")
    print(f"    breach the account is dead and keeps the equity it had at that instant.")

    print(f"\n  {'Size':<10} {'1 stop':>9} {'P(survive)':>11} {'P(pass $3k)':>12} "
          f"{'median':>9} {'95th pct':>10} {'max obs':>10}")
    print(f"  {'-' * 74}")
    for r in survivors:
        print(f"  {r.label:<10} {r.stop_out_dollars:>9,.0f} {r.p_survive:>11.1%} "
              f"{r.p_reach_target:>12.1%} {r.median_return_pct:>8.1f}% "
              f"{r.p95_return_pct:>9.1f}% {r.max_return_pct:>9.1f}%")

    best = prop.get("best")
    if best is not None:
        print(f"\n  Best surviving size ({prop['best_note']}): {best.label}")
        print(f"    P(survive the year):        {best.p_survive:.1%}")
        print(f"    P(reach ${rules.profit_target:,.0f} target):    {best.p_reach_target:.1%}")
        print(f"    Median annual return:       {best.median_return_pct:.1f}% "
              f"(${best.median_return_pct / 100 * rules.account_size:,.0f})")
        print(f"    95th percentile:            {best.p95_return_pct:.1f}%")
        print(f"    Max observed:               {best.max_return_pct:.1f}%")
    else:
        print(f"\n  No size survives the daily loss limit filter.")

    print(f"\n  Step 3: replay the realized OOS days once, in order, at each surviving")
    print(f"    size. This is history, not resampling.")
    print(f"\n  {'Size':<10} {'Outcome':<39} {'Final equity':>13} {'Worst day':>11}")
    print(f"  {'-' * 76}")
    for row in prop["replays"]:
        if row["survived"]:
            outcome_text = (
                "survived; $3k target reached"
                if row["reached_target"]
                else "survived; target not reached"
            )
        else:
            outcome_text = f"KILLED {row['breach_day'].date()} ({row['breach_reason']})"
        print(f"  {row['label']:<10} {outcome_text:<39} ${row['final_equity']:>12,.0f} "
              f"${row['worst_day_dollars']:>10,.0f}")

    ceiling = prop["ceiling_pct"]
    max_dpp = rules.micro_point_value * rules.max_micro_contracts
    print(f"\n  ARITHMETIC CEILING IMPOSED BY THE {rules.max_micro_contracts}-MICRO CAP")
    print(f"    {rules.max_micro_contracts} Micro on every trade (${max_dpp:.0f}/pt), "
          f"{prop['trades_per_day']:.2f} trades/day, "
          f"{rules.trading_days_per_year} days,")
    print(f"    the OOS EV of {prop['ev_points']:.2f} pts earned on every trade, zero "
          f"losing days, zero")
    print(f"    breaches, no compounding (the contract cap does not grow with equity).")
    print(f"    Ceiling = {rules.trading_days_per_year} x {prop['trades_per_day']:.2f} x "
          f"{prop['ev_points']:.2f} x ${max_dpp:.0f} = "
          f"${ceiling / 100 * rules.account_size:,.0f} = {ceiling:.1f}% / year")
    if prop["max_micro_eliminated"]:
        print(f"    NOTE: {rules.max_micro_contracts} Micro is itself eliminated by the "
              f"daily loss limit, so this")
        print(f"    ceiling is not even legally reachable.")
    print(f"\n    Claimed figure:  {CLAIMED_ANNUAL_RETURN_PCT:,.0f}%")
    print(f"    Hard ceiling:    {ceiling:.1f}%")
    if ceiling > 0:
        print(f"    The claim is {CLAIMED_ANNUAL_RETURN_PCT / ceiling:,.0f}x the ceiling "
              f"and is unreachable.")
    else:
        print(f"    The ceiling is not positive, so the claim is unreachable.")


def print_cot_section(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print the COT Index result or say plainly that it is missing."""
    cot = outcome.cot
    print(f"\n{'=' * 76}")
    print("  COMMITMENTS OF TRADERS (optional stretch)")
    print(f"{'=' * 76}")

    if not cot.get("available"):
        print(f"\n  ABSENT. {cot.get('reason', 'Not attempted.')}")
        print("  COT is arguably Williams' strongest input - he used the commercial")
        print("  net position as the primary directional bias and treated price")
        print("  patterns as timing. This replication does not contain it, so it is")
        print("  a replication of Williams' timing tools only.")
        return

    print(f"\n  Source: {cot['source']}")
    print(f"  Contract: {cot['contract']}")
    print(f"  Weekly reports: {cot['n_reports']} "
          f"({cot['first_report']} to {cot['last_report']})")
    print(f"  COT Index = percentile rank of commercial net position over a "
          f"{cot['lookback_weeks']}-week")
    print(f"  ({cot['lookback_weeks'] / 52:.0f}-year) lookback, as Williams defines it. "
          f"Each daily bar uses the most")
    print(f"  recent report published strictly before that day, so nothing is forward-looking.")

    print(f"\n  Tested on TRAIN only, as a directional bias filter on the locked "
          f"signal set:")
    print(f"  {'Variant':<40} {'Trades':>7} {'Net pts':>10} {'Net $':>12}")
    print(f"  {'-' * 71}")
    for label, n_trades, points in cot["variants"]:
        print(f"  {label:<40} {n_trades:>7} {points:>10.1f} {points * point_value:>12,.0f}")
    print(f"\n  {cot['verdict']}")


def print_final_summary(outcome: WilliamsOutcome, point_value: float) -> None:
    """Print the single consolidated summary."""
    print(f"\n{'=' * 76}")
    print("  FINAL SUMMARY - WILLIAMS DAILY SYSTEM ON NQ")
    print(f"{'=' * 76}")

    print(f"\n  DATA")
    print(f"    {outcome.daily_days} daily RTH bars, "
          f"{outcome.date_range[0]} to {outcome.date_range[1]}, resampled from the")
    print(f"    1-minute file. Split: chronological 50 / embargo 5 / 20 / embargo 5 / 30.")

    print(f"\n  COMPONENT STANDALONE TRAIN PERFORMANCE")
    for score in outcome.component_scores:
        verdict = "POSITIVE" if score.total_pnl_points > 0 else "NEGATIVE"
        print(f"    {score.name:<6} {score.n_trades:>4} trades  "
              f"{score.total_pnl_points:>9.1f} pts  "
              f"${score.total_pnl_points * point_value:>10,.0f}  "
              f"EV {score.ev_per_trade_points:>6.2f}  {verdict}")

    print(f"\n  COMPONENTS IN THE FINAL STRATEGY")
    print(f"    {', '.join(outcome.chosen_components)}"
          f"{' + TDOM filter' if outcome.chosen_tdom_filter else ''}")
    print(f"    {outcome.combination_reason}")

    if outcome.stopped_on_train_failure:
        print(f"\n  HEADLINE FINDING")
        print(f"    0 of {len(outcome.train_scores)} grid combinations are profitable on "
              f"TRAIN.")
        print(f"    The pipeline stopped there. No validation or OOS numbers are shown,")
        print(f"    because a strategy the training period rejects cannot be rescued by")
        print(f"    a favourable later period.")
        print(f"{'=' * 76}")
        return

    print(f"\n  LOCKED PARAMETERS")
    for key in ("components", "gsv_lookback", "gsv_multiplier", "stop_points",
                "target_points", "max_hold_days", "tdom_filter"):
        if key in outcome.locked:
            print(f"    {key:<20} {outcome.locked[key]}")
    print(f"    {'grid size':<20} {len(outcome.train_scores)} combinations "
          f"(cap {MAX_GRID_COMBINATIONS}); "
          f"{outcome.train_profitable_combos} profitable on train")
    print(f"    {'train evaluations':<20} {outcome.total_train_evaluations} in total "
          f"(components, combinations, grid)")

    print(f"\n  TRAIN / VALIDATION / TRUE OOS")
    print(f"    {'':<24} {'Train':>12} {'Validation':>12} {'True OOS':>12}")
    for key, label, kind in (
        ("total_trades", "Trades", "int"),
        ("win_rate", "Win rate", "pct"),
        ("expected_value", "EV/trade (pts)", "f2"),
        ("profit_factor", "Profit factor", "f2"),
        ("max_consecutive_losers", "Max consec losers", "int"),
        ("total_return", "Net (pts)", "f1"),
    ):
        values = []
        for metrics in (outcome.train_metrics, outcome.validation_metrics, outcome.oos_metrics):
            value = metrics.get(key, 0)
            if kind == "int":
                values.append(f"{int(value)}")
            elif kind == "pct":
                values.append(f"{value:.1%}")
            elif kind == "f1":
                values.append(f"{value:.1f}")
            else:
                values.append(f"{value:.2f}")
        print(f"    {label:<24} {values[0]:>12} {values[1]:>12} {values[2]:>12}")

    wf = outcome.walk_forward
    print(f"\n  WALK-FORWARD CONSISTENCY")
    if wf.get("windows"):
        print(f"    {wf['profitable']} of {wf['total']} windows profitable "
              f"({wf['consistency']:.1%}); combined "
              f"{wf['total_points']:.1f} pts "
              f"(${wf['total_points'] * point_value:,.0f})")
        print(f"    {wf['distinct_param_sets']} distinct parameter sets chosen across "
              f"{wf['total']} windows")
    else:
        print(f"    Not run")

    stats = outcome.gap_stats
    print(f"\n  GAP-FILL STATISTICS (OOS)")
    print(f"    {stats['gap_fill_losers']} of {stats['losing_trades']} losing trades "
          f"({stats['gap_fill_share_of_losers']:.1%}) were gap fills")
    print(f"    Average gap fill {stats['avg_gap_slippage_points']:.1f} pts worse than the "
          f"stop (${stats['avg_gap_slippage_points'] * point_value:,.0f}); worst "
          f"{stats['worst_gap_slippage_points']:.1f} pts "
          f"(${stats['worst_gap_slippage_points'] * point_value:,.0f})")

    prop = outcome.prop
    rules: PropFirmRules = prop["rules"]
    best = prop.get("best")
    print(f"\n  FUNDEDNEXT 50K")
    micro_kept = [r for r in prop["size_results"]
                  if r.contract_type == "Micro" and not r.eliminated]
    mini_kept = [r for r in prop["size_results"]
                 if r.contract_type == "Mini" and not r.eliminated]
    print(f"    Locked stop {prop['stop_points']:.0f} pts -> legal sizes: "
          f"{len(micro_kept)} Micro sizes, {len(mini_kept)} Mini sizes")
    print(f"    Eliminated: Micro "
          f"{len([r for r in prop['size_results'] if r.contract_type == 'Micro' and r.eliminated])}"
          f" sizes, Mini "
          f"{len([r for r in prop['size_results'] if r.contract_type == 'Mini' and r.eliminated])}"
          f" sizes (one stop-out alone exceeds ${rules.daily_loss_limit:,.0f})")
    print(f"    Gap-aware cap: at most {prop['gap_aware_max_contracts']['micro']} Micro "
          f"keeps the worst observed day inside the limit")
    if best is not None:
        print(f"    Best surviving size: {best.label} - P(survive) {best.p_survive:.1%}, "
              f"P(pass) {best.p_reach_target:.1%}")
        print(f"    Median annual return {best.median_return_pct:.1f}%, "
              f"95th pct {best.p95_return_pct:.1f}%, max {best.max_return_pct:.1f}%")
    else:
        print(f"    No size survives the daily loss limit")
    print(f"    Arithmetic ceiling {prop['ceiling_pct']:.1f}% / year vs claimed "
          f"{CLAIMED_ANNUAL_RETURN_PCT:,.0f}%")

    print(f"\n  VERDICT")
    checks = _verdict_checks(outcome)
    for label, passed, detail in checks:
        print(f"    {'PASS' if passed else 'FAIL'}  {label:<38} {detail}")
    passes = sum(1 for _, ok, _ in checks if ok)
    print(f"\n    {passes}/{len(checks)} checks pass.")

    for line in _verdict_prose(outcome, checks):
        print(f"    {line}")

    print(f"\n  Every number above comes from the bar-by-bar daily simulation or from a")
    print(f"  Monte Carlo that replays those simulated days one at a time. No P&L was")
    print(f"  capped, smoothed or adjusted after the fact.")
    print(f"{'=' * 76}")


def _verdict_checks(outcome: WilliamsOutcome) -> list[tuple[str, bool, str]]:
    """Build the pass/fail checklist for the verdict."""
    wf = outcome.walk_forward
    prop = outcome.prop
    best = prop.get("best")
    train_ev = outcome.train_metrics.get("expected_value", 0.0)
    val_ev = outcome.validation_metrics.get("expected_value", 0.0)
    oos_ev = outcome.oos_metrics.get("expected_value", 0.0)

    return [
        ("Any grid combo works on train", outcome.train_profitable_combos > 0,
         f"{outcome.train_profitable_combos} of {len(outcome.train_scores)}"),
        ("Train profitable", train_ev > 0, f"EV {train_ev:+.2f} pts/trade"),
        ("Validation profitable", val_ev > 0, f"EV {val_ev:+.2f} pts/trade"),
        ("True OOS profitable", oos_ev > 0, f"EV {oos_ev:+.2f} pts/trade"),
        ("Walk-forward > 50% of windows",
         bool(wf.get("windows")) and wf.get("consistency", 0.0) > 0.50,
         f"{wf.get('consistency', 0.0):.0%}" if wf.get("windows") else "not run"),
        ("A legal FN size survives the year",
         best is not None and best.p_survive >= 0.90,
         f"{best.label} at {best.p_survive:.0%}" if best else "none"),
        ("FN profit target reachable",
         best is not None and best.p_reach_target >= 0.50,
         f"{best.p_reach_target:.0%}" if best else "n/a"),
    ]


def _verdict_prose(outcome: WilliamsOutcome, checks: list[tuple[str, bool, str]]) -> list[str]:
    """Write the plain-language verdict lines."""
    lines: list[str] = []
    train_ok = outcome.train_metrics.get("expected_value", 0.0) > 0
    oos_ok = outcome.oos_metrics.get("expected_value", 0.0) > 0
    wf_ok = outcome.walk_forward.get("consistency", 0.0) > 0.50
    best = outcome.prop.get("best")
    tradeable = bool(best) and best.p_survive >= 0.90 and oos_ok

    lines.append("")
    lines.append("Does this replicate Williams' methodology? Partly. GSV, Oops!, Smash")
    lines.append("Day and TDOM seasonality are all implemented on daily bars with")
    lines.append("stop-order entries and multi-day holds, which is the shape of his")
    lines.append("systems. What is missing is the part he leaned on hardest: the COT")
    if outcome.cot.get("available"):
        lines.append("Index was fetched and measured, but it did not improve train P&L, so")
        lines.append("it is reported and NOT part of the traded system. Williams used")
        lines.append("commercial positioning as the primary bias and price patterns as")
        lines.append("timing, so what is validated here is his timing tools alone.")
    else:
        lines.append("Index, which is absent here (see the COT section). Williams used")
        lines.append("commercial positioning as the primary bias and price patterns as")
        lines.append("timing, so what is validated here is his timing tools alone.")

    lines.append("")
    if not train_ok:
        lines.append("Is there an edge? No. The locked parameters do not even make money")
        lines.append("on train, so nothing downstream counts.")
    elif not oos_ok:
        lines.append("Is there an edge? Not demonstrated. Train and validation are")
        lines.append("positive but true OOS is not, which is the classic signature of a")
        lines.append("parameter set fitted to the earlier period.")
    elif not wf_ok:
        lines.append("Is there an edge? Unproven. OOS is positive but the walk-forward")
        lines.append("windows disagree, so the result depends on which period you pick.")
    else:
        lines.append("Is there an edge? The measurements are consistently positive across")
        lines.append("train, validation, OOS and the majority of walk-forward windows.")
        lines.append("That is the strongest statement this data supports; it is not proof.")

    lines.append("")
    replays = outcome.prop.get("replays", [])
    survived_history = [r["label"] for r in replays if r["survived"]]

    if tradeable:
        lines.append(f"Is it tradeable under FN 50K rules? Marginally, at {best.label} only.")
        lines.append("The binding constraint is the overnight gap: a stop cannot be")
        lines.append("honoured through a gap, and a single gap day at any meaningful size")
        lines.append("breaches the $1,000 daily limit outright.")
    else:
        results = outcome.prop.get("size_results", [])
        eliminated = sum(1 for r in results if r.eliminated)
        survivors = [r for r in results if not r.eliminated]
        lines.append("Is it tradeable under FN 50K rules? No. A daily-bar system holds")
        lines.append("overnight, the overnight move is filled at the open rather than at")
        lines.append("the stop, and the $1,000 daily limit is a one-strike rule.")
        lines.append(f"{eliminated} of the {len(results)} available sizes are eliminated "
                     f"before any simulation")
        lines.append(f"runs, because one stop-out alone breaches the limit. Of the "
                     f"{len(survivors)} that")
        if best is not None:
            lines.append(f"survive, the best is {best.label}, and even there the account "
                         f"breaches a rule in")
            lines.append(f"{1 - best.p_survive:.0%} of simulated years for a median return "
                         f"of {best.median_return_pct:.1f}%.")
        if survived_history:
            lines.append(f"The realized OOS days do survive at "
                         f"{', '.join(survived_history)} and reach the")
            lines.append("$3,000 target, but that is a single path. The resampled")
            lines.append("distribution is the honest estimate of what to expect.")

    return lines


# -------------------------------------------------------------------- pipeline


def _measure_components(
    engine: DailyBacktestEngine,
    df: pd.DataFrame,
    train: Segment,
    tdom_table: dict[int, float],
) -> tuple[list[ComponentScore], ComponentScore]:
    """Measure all four components standalone on train, plus the inverted-GSV diagnostic.

    Args:
        engine: Daily engine.
        df: Full daily frame.
        train: Train segment.
        tdom_table: TDOM table fitted on train.

    Returns:
        Tuple of (component scores in declaration order, inverted GSV score).
    """
    scores: list[ComponentScore] = []
    for component in ("gsv", "oops", "smash", "tdom"):
        params = {
            **STANDALONE_EXITS,
            "components": (component,),
            "tdom_filter": False,
            "tdom_bias": tdom_table,
            "gsv_inverted": False,
            "gsv_lookback": 5,
            "gsv_multiplier": 0.8,
            "smash_lookback": 5,
        }
        scores.append(_score_from_result(component, _run_once(engine, df, train, params)))

    inverted = _score_from_result(
        "gsv",
        _run_once(
            engine,
            df,
            train,
            {**STANDALONE_EXITS, "components": ("gsv",), "gsv_inverted": True,
             "tdom_filter": False, "gsv_lookback": 5, "gsv_multiplier": 0.8},
        ),
    )
    engine.strategy.params["gsv_inverted"] = False
    return scores, inverted


def _choose_combination(
    engine: DailyBacktestEngine,
    df: pd.DataFrame,
    train: Segment,
    component_scores: list[ComponentScore],
    tdom_table: dict[int, float],
) -> tuple[tuple[str, ...], bool, str, list[tuple[str, float, int]]]:
    """Pick the component combination using TRAIN results only.

    Only components that are profitable standalone are candidates. TDOM can
    only enter as a filter. Candidate entry sets are cumulative prefixes of the
    positive components ordered by standalone P&L; each candidate is measured on
    train with and without the TDOM filter, and the best train P&L wins.

    Args:
        engine: Daily engine.
        df: Full daily frame.
        train: Train segment.
        component_scores: Standalone scores.
        tdom_table: TDOM table fitted on train.

    Returns:
        Tuple of (chosen entry components, chosen tdom_filter flag, reason,
        table of all candidates measured).
    """
    by_name = {s.name: s for s in component_scores}
    entry_candidates = [
        s.name
        for s in sorted(component_scores, key=lambda s: s.total_pnl_points, reverse=True)
        if s.total_pnl_points > 0 and s.name != "tdom"
    ]
    tdom_positive = by_name["tdom"].total_pnl_points > 0

    if not entry_candidates:
        return ("gsv",), False, (
            "No entry component is profitable standalone on train; GSV is kept as "
            "the core so the failure is reported against the intended system."
        ), []

    filter_options = [False, True] if tdom_positive else [False]

    table: list[tuple[str, float, int]] = []
    best: tuple[tuple[str, ...], bool, float] | None = None
    for size in range(1, len(entry_candidates) + 1):
        components = tuple(entry_candidates[:size])
        for use_filter in filter_options:
            params = {
                **STANDALONE_EXITS,
                "components": components,
                "tdom_filter": use_filter,
                "tdom_bias": tdom_table,
                "gsv_inverted": False,
                "gsv_lookback": 5,
                "gsv_multiplier": 0.8,
                "smash_lookback": 5,
            }
            result = _run_once(engine, df, train, params)
            points = result.total_pnl_points
            label = " + ".join(components) + (" + TDOM filter" if use_filter else "")
            table.append((label, points, len(result.trades)))
            if best is None or points > best[2]:
                best = (components, use_filter, points)

    components, use_filter, points = best
    excluded = [s.name for s in component_scores if s.total_pnl_points <= 0]
    reason_parts = [
        f"{', '.join(components)}"
        f"{' with the TDOM filter' if use_filter else ''} had the best TRAIN P&L "
        f"({points:.0f} pts) of the {len(table)} candidate combinations."
    ]
    if excluded:
        reason_parts.append(
            f"Excluded as unprofitable standalone on train: {', '.join(excluded)}."
        )
    dropped = [c for c in entry_candidates if c not in components]
    if dropped:
        reason_parts.append(
            f"{', '.join(dropped)} is profitable standalone but degrades the "
            f"combination: its stop order sits closer to the open than the GSV "
            f"breakout level, so it pre-empts the breakout entry rather than adding "
            f"to it."
        )
    if tdom_positive and not use_filter:
        reason_parts.append(
            "The TDOM filter is profitable standalone but reduces train P&L as a "
            "filter, so it is not used."
        )

    return components, use_filter, " ".join(reason_parts), table


def _run_walk_forward(
    engine: DailyBacktestEngine,
    df: pd.DataFrame,
    grid: dict[str, list],
    base_params: dict[str, Any],
    cost_model: CostModel,
    wf_config: dict,
    point_value: float,
    target_r_multiple: float,
) -> dict[str, Any]:
    """Walk-forward over windows generated by the existing WalkForwardAnalyzer.

    Args:
        engine: Daily engine.
        df: Full daily frame.
        grid: Same capped grid used for the main selection.
        base_params: Locked non-grid parameters (components, tdom table, etc).
        cost_model: Cost model (only needed to construct the analyzer).
        wf_config: Section with train_window_months / test_window_months / step_months.
        point_value: Dollar value per point.
        target_r_multiple: Fixed reward:risk multiple.

    Returns:
        Dict summarizing the windows.
    """
    train_months = int(wf_config.get("train_window_months", 18))
    test_months = int(wf_config.get("test_window_months", 6))
    step_months = int(wf_config.get("step_months", 6))

    analyzer = WalkForwardAnalyzer(
        strategy=engine.strategy,
        cost_model=cost_model,
        train_window_months=train_months,
        test_window_months=test_months,
        step_months=step_months,
        point_value=point_value,
    )
    windows = analyzer._generate_windows(df)

    combos = build_daily_combos(grid, target_r_multiple=target_r_multiple)
    rows: list[dict[str, Any]] = []
    total_points = 0.0
    param_sets: set[tuple] = set()

    for idx, (train_df, test_df) in enumerate(windows, start=1):
        train_segment = Segment(
            "wf_train",
            df.index.get_loc(train_df.index[0]),
            df.index.get_loc(train_df.index[-1]) + 1,
        )
        test_segment = Segment(
            "wf_test",
            df.index.get_loc(test_df.index[0]),
            df.index.get_loc(test_df.index[-1]) + 1,
        )

        # Fit the TDOM table on this window's train rows only.
        window_params = dict(base_params)
        window_params["tdom_bias"] = tdom_bias_table(
            df.iloc[train_segment.start_pos : train_segment.end_pos]
        )

        signal_cache: dict = {}
        best: DailyComboScore | None = None
        for combo in combos:
            score = evaluate_daily_params(
                engine, df, train_segment, {**window_params, **combo}, signal_cache
            )
            score.result = None
            if best is None or score.score > best.score:
                best = score

        test_score = evaluate_daily_params(
            engine, df, test_segment, {**window_params, **best.params}, signal_cache
        )
        total_points += test_score.total_pnl_points
        param_sets.add(
            (best.params["gsv_lookback"], best.params["gsv_multiplier"],
             best.params["stop_points"], best.params["max_hold_days"])
        )

        rows.append(
            {
                "idx": idx,
                "train_end": train_df.index[-1].date(),
                "test_start": test_df.index[0].date(),
                "test_end": test_df.index[-1].date(),
                "trades": test_score.n_trades,
                "net_points": test_score.total_pnl_points,
                "params": best.params,
            }
        )

    profitable = sum(1 for row in rows if row["net_points"] > 0)
    return {
        "windows": rows,
        "total": len(rows),
        "profitable": profitable,
        "consistency": profitable / len(rows) if rows else 0.0,
        "total_points": total_points,
        "distinct_param_sets": len(param_sets),
        "train_months": train_months,
        "test_months": test_months,
        "step_months": step_months,
    }


def _run_prop_firm(
    config: dict,
    oos_result: DailySimResult,
    stop_points: float,
    point_value: float,
) -> dict[str, Any]:
    """FundedNext sizing: elimination filter, Monte Carlo, historical replay, ceiling.

    Args:
        config: Full config dict.
        oos_result: True OOS simulation result.
        stop_points: Locked stop distance in points.
        point_value: Dollar value per point of the backtested contract.

    Returns:
        Dict consumed by the printing functions.
    """
    rules = PropFirmRules.from_config(config)
    prop_cfg = config.get("prop_firm", {})
    n_sims = int(prop_cfg.get("simulations", 10_000))
    seed = int(prop_cfg.get("seed", 42))

    daily_pnl = oos_result.daily_pnl_points
    units = daily_pnl_units(daily_pnl)

    size_results = evaluate_sizes(
        trades=units,
        trading_days=len(daily_pnl),
        stop_points=stop_points,
        rules=rules,
        n_sims=n_sims,
        seed=seed,
    )
    survivors = [r for r in size_results if not r.eliminated]

    robust = [r for r in survivors if r.p_survive >= 0.90]
    pool = robust if robust else survivors
    best = max(pool, key=lambda r: r.median_return_pct) if pool else None
    best_note = "P(survive) >= 90%" if robust else "no size survives 90% of years"

    worst_day_points = float(daily_pnl.min()) if len(daily_pnl) else 0.0
    worst_day_loss = abs(worst_day_points)
    gap_aware = {
        "micro": int(rules.daily_loss_limit // (worst_day_loss * rules.micro_point_value))
        if worst_day_loss else rules.max_micro_contracts,
        "mini": int(rules.daily_loss_limit // (worst_day_loss * rules.mini_point_value))
        if worst_day_loss else rules.max_mini_contracts,
    }
    gap_aware["micro"] = min(gap_aware["micro"], rules.max_micro_contracts)
    gap_aware["mini"] = min(gap_aware["mini"], rules.max_mini_contracts)
    gap_aware["micro_dollars"] = worst_day_loss * rules.micro_point_value * max(
        gap_aware["micro"], 1
    )

    replays = []
    for size in survivors:
        replay = replay_prop_account(
            daily_pnl_points=daily_pnl,
            dollars_per_point=size.dollars_per_point,
            account_size=rules.account_size,
            daily_loss_limit=rules.daily_loss_limit,
            equity_floor=rules.equity_floor,
            profit_target=rules.profit_target,
        )
        replay["label"] = size.label
        replays.append(replay)

    trades = oos_result.trades
    ev_points = sum(t.pnl_net for t in trades) / len(trades) if trades else 0.0
    trades_per_day = len(trades) / len(daily_pnl) if len(daily_pnl) else 0.0
    ceiling = arithmetic_ceiling_pct(ev_points, trades_per_day, rules)

    return {
        "rules": rules,
        "size_results": size_results,
        "best": best,
        "best_note": best_note,
        "replays": replays,
        "n_sims": n_sims,
        "stop_points": stop_points,
        "oos_trades": len(trades),
        "oos_days": len(daily_pnl),
        "ev_points": ev_points,
        "trades_per_day": trades_per_day,
        "ceiling_pct": ceiling,
        "worst_day_points": worst_day_points,
        "gap_aware_max_contracts": gap_aware,
        "max_micro_eliminated": any(
            r.eliminated and r.contract_type == "Micro"
            and r.contracts == rules.max_micro_contracts
            for r in size_results
        ),
    }


def run_williams_pipeline(config: dict) -> WilliamsOutcome:
    """Run the whole Williams daily-bar study and print it.

    Args:
        config: Full configuration dict.

    Returns:
        WilliamsOutcome with everything that was measured.
    """
    started = time.time()
    project_root = get_project_root()

    williams_cfg = config.get("williams", {})
    costs_cfg = config.get("costs", {})
    point_value = float(costs_cfg.get("point_value", 20.0))
    selection_cfg = williams_cfg.get("selection", {})
    split_cfg = williams_cfg.get("split", {})
    target_r_multiple = float(williams_cfg.get("target_r_multiple", 2.0))

    print("=" * 76)
    print("  LARRY WILLIAMS DAILY SYSTEM ON NQ - FundedNext 50K feasibility")
    print("=" * 76)

    # ---------------------------------------------------- step 1: daily bars
    print("\n[1/9] Building daily RTH bars from the 1-minute file...")
    minute_file = project_root / williams_cfg.get(
        "minute_file", "data/NQ_1min_clean_2021_2026.parquet"
    )
    daily_file = project_root / williams_cfg.get("daily_file", "data/NQ_daily_rth.parquet")
    daily = load_daily_rth_bars(
        minute_path=minute_file,
        cache_path=daily_file,
        force_rebuild=bool(williams_cfg.get("force_rebuild_daily", False)),
    )
    gap_series = overnight_gap_points(daily)
    print(f"  {len(daily)} daily RTH bars (09:30-16:00 ET), "
          f"{daily.index[0].date()} to {daily.index[-1].date()}")
    print(f"  Cached to {daily_file.relative_to(project_root)}")
    print(f"  Median overnight gap {gap_series.abs().median():.1f} pts, "
          f"largest {gap_series.abs().max():.1f} pts")

    print("\n  Loading 1-minute session paths so intraday fill order is resolved "
          "rather than assumed...")
    intraday = IntradayPaths(pd.read_parquet(minute_file))

    # ---------------------------------------------------------- step 2: split
    print("\n[2/9] Building the embargoed chronological split...")
    segments = build_embargo_split(
        daily,
        train_fraction=float(split_cfg.get("train_fraction", 0.50)),
        validation_fraction=float(split_cfg.get("validation_fraction", 0.20)),
        embargo_days=int(split_cfg.get("embargo_days", 5)),
    )
    print_embargo_audit(daily, segments)
    train = segment_by_name(segments, "train")
    validation = segment_by_name(segments, "validation")
    oos = segment_by_name(segments, "oos")

    outcome = WilliamsOutcome(
        daily_days=len(daily),
        date_range=(daily.index[0].date(), daily.index[-1].date()),
        segments=segments,
    )

    cost_model = CostModel.from_config(costs_cfg)
    strategy = WilliamsStrategy()
    engine = DailyBacktestEngine(
        strategy=strategy, cost_model=cost_model, intraday=intraday, close_at_end=True
    )

    # TDOM is fitted on TRAIN rows only and then frozen.
    tdom_table = tdom_bias_table(daily.iloc[train.start_pos : train.end_pos])

    # ------------------------------------------------- step 3: components
    print("\n[3/9] Measuring each component standalone on TRAIN...")
    component_scores, inverted = _measure_components(engine, daily, train, tdom_table)
    outcome.component_scores = component_scores
    outcome.inverted_gsv = inverted
    print_component_table(outcome, point_value)

    print(f"\n  TDOM table fitted on TRAIN only "
          f"({len(tdom_table)} trading-day indices with enough observations):")
    ordered = sorted(tdom_table.items())
    for start in range(0, len(ordered), 6):
        chunk = ordered[start : start + 6]
        print("    " + "  ".join(f"day {day:>2}: {value:>+7.1f} pts" for day, value in chunk))

    # ------------------------------------------------ step 4: combination
    print("\n[4/9] Deciding which components to combine, on TRAIN...")
    components, tdom_filter, reason, combo_table = _choose_combination(
        engine, daily, train, component_scores, tdom_table
    )
    outcome.chosen_components = components
    outcome.chosen_tdom_filter = tdom_filter
    outcome.combination_reason = reason
    outcome.combination_scores = combo_table
    print_combination_choice(outcome, point_value)

    base_params = {
        "components": components,
        "tdom_filter": tdom_filter,
        "tdom_bias": tdom_table,
        "gsv_inverted": False,
        "smash_lookback": 5,
        "trailing_stop_points": None,
    }

    # --------------------------------------------------- step 5: selection
    print("\n[5/9] Scoring the grid on TRAIN, picking on VALIDATION...")
    grid = {
        "gsv_lookback": list(selection_cfg.get("gsv_lookback", [3, 5, 10])),
        "gsv_multiplier": list(selection_cfg.get("gsv_multiplier", [0.6, 1.0])),
        "stop_points": list(selection_cfg.get("stop_points", [50, 75])),
        "max_hold_days": list(selection_cfg.get("max_hold_days", [1, 3, 5])),
    }
    top_n = int(selection_cfg.get("top_n_for_validation", 3))

    strategy.params.update(base_params)
    locked, train_scores, validation_scores = select_daily_parameters(
        engine=engine,
        df=daily,
        train=train,
        validation=validation,
        grid=grid,
        min_trades=int(selection_cfg.get("min_trades", 20)),
        top_n=top_n,
        target_r_multiple=target_r_multiple,
    )
    locked = {**base_params, **locked}
    outcome.locked = locked
    outcome.train_scores = train_scores
    outcome.validation_scores = validation_scores
    outcome.train_profitable_combos = sum(1 for s in train_scores if s.score > 0)
    outcome.total_train_evaluations = (
        len(component_scores) + 1 + len(combo_table) + len(train_scores)
    )
    print_selection_report(outcome, grid, point_value, top_n)

    if outcome.train_profitable_combos == 0:
        outcome.stopped_on_train_failure = True
        print(f"\n{'!' * 76}")
        print("  HEADLINE FINDING: 0 of "
              f"{len(train_scores)} grid combinations are profitable on TRAIN.")
        print("  Stopping here. Validation and OOS are not evaluated, because a")
        print("  strategy the training period rejects cannot be validated by a")
        print("  luckier later period.")
        print(f"{'!' * 76}")
        print_final_summary(outcome, point_value)
        return outcome

    # ------------------------------------------ step 6: locked run, incl. OOS
    print("\n[6/9] Running the locked parameters on train, validation and OOS...")
    train_result = _run_once(engine, daily, train, locked)
    validation_result = _run_once(engine, daily, validation, locked)
    oos_result = _run_once(engine, daily, oos, locked)

    outcome.train_metrics = compute_all_metrics(train_result.to_backtest_result(), point_value)
    outcome.validation_metrics = compute_all_metrics(
        validation_result.to_backtest_result(), point_value
    )
    outcome.oos_metrics = compute_all_metrics(oos_result.to_backtest_result(), point_value)
    print_split_comparison(outcome, point_value)

    # ------------------------------------------------ step 7: walk-forward
    print("\n[7/9] Walk-forward...")
    outcome.walk_forward = _run_walk_forward(
        engine=engine,
        df=daily,
        grid=grid,
        base_params=base_params,
        cost_model=cost_model,
        wf_config=williams_cfg.get("walk_forward", {}),
        point_value=point_value,
        target_r_multiple=target_r_multiple,
    )
    print_walk_forward(outcome, point_value)

    # Re-apply the locked parameters: the walk-forward mutated them.
    strategy.params.update(locked)

    # -------------------------------------------------- step 8: gap analysis
    print("\n[8/9] Gap-fill statistics and FundedNext sizing...")
    outcome.gap_stats = gap_fill_statistics(oos_result.trades)
    outcome.prop = _run_prop_firm(
        config=config,
        oos_result=oos_result,
        stop_points=float(locked["stop_points"]),
        point_value=point_value,
    )
    print_gap_analysis(outcome, point_value, gap_series)
    print_prop_firm(outcome, point_value)

    # --------------------------------------------------------- step 9: COT
    print("\n[9/9] Commitments of Traders (optional stretch)...")
    outcome.cot = _try_cot(config, engine, daily, train, locked, point_value)
    print_cot_section(outcome, point_value)

    # Trade logs
    results_dir = project_root / "results"
    save_daily_trade_log(oos_result, str(results_dir / "williams_oos_trades.csv"), point_value)
    save_daily_trade_log(train_result, str(results_dir / "williams_train_trades.csv"), point_value)
    oos_result.daily_pnl_points.rename("pnl_points").to_csv(
        results_dir / "williams_oos_daily_pnl.csv"
    )

    print_final_summary(outcome, point_value)
    print(f"\n  Williams pipeline time: {time.time() - started:.0f}s")
    return outcome


def _try_cot(
    config: dict,
    engine: DailyBacktestEngine,
    daily: pd.DataFrame,
    train: Segment,
    locked: dict[str, Any],
    point_value: float,
) -> dict[str, Any]:
    """Attempt the COT Index bias filter, reporting cleanly if it is unavailable.

    Args:
        config: Full config dict.
        engine: Daily engine.
        daily: Daily bar frame.
        train: Train segment.
        locked: Locked parameters.
        point_value: Dollar value per point.

    Returns:
        Dict consumed by :func:`print_cot_section`.
    """
    cot_cfg = config.get("williams", {}).get("cot", {})
    if not cot_cfg.get("enabled", True):
        return {"available": False, "reason": "Disabled in config."}

    try:
        from src.data.cot import COT_CONTRACT, cot_index_daily, load_cot_index
    except ImportError as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"COT module unavailable: {exc}"}

    try:
        weekly = load_cot_index(
            cache_path=get_project_root() / cot_cfg.get("cache_file", "data/cot_nq.parquet"),
            start_year=int(cot_cfg.get("start_year", 2018)),
            end_year=int(cot_cfg.get("end_year", 2026)),
            lookback_weeks=int(cot_cfg.get("lookback_weeks", 156)),
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": (f"The CFTC fetch did not complete cleanly ({type(exc).__name__}: "
                       f"{exc}), so COT is excluded rather than guessed at."),
        }

    if weekly is None or weekly.empty:
        return {"available": False, "reason": "No COT rows were parsed for the contract."}

    index_series = cot_index_daily(weekly, daily.index)
    coverage = float(index_series.notna().mean())
    if coverage < 0.9:
        return {
            "available": False,
            "reason": (f"COT covered only {coverage:.0%} of the daily bars, too little to "
                       f"test honestly."),
        }

    bullish = float(cot_cfg.get("bullish_threshold", 80.0))
    bearish = float(cot_cfg.get("bearish_threshold", 20.0))

    variants: list[tuple[str, int, float]] = []
    baseline = _run_once(engine, daily, train, locked)
    variants.append(("no COT filter (locked signal set)", len(baseline.trades),
                     baseline.total_pnl_points))

    signals = engine.strategy.generate_signals(daily)
    for label, long_ok, short_ok in (
        (f"longs when COT index > {bullish:.0f}, shorts when < {bearish:.0f}",
         index_series > bullish, index_series < bearish),
        ("longs when COT index > 50, shorts when < 50",
         index_series > 50.0, index_series < 50.0),
    ):
        filtered = signals.copy()
        filtered["long_trigger"] = filtered["long_trigger"].where(long_ok.fillna(False))
        filtered["short_trigger"] = filtered["short_trigger"].where(short_ok.fillna(False))
        result = engine.run(daily, filtered, train.start_pos, train.end_pos)
        variants.append((label, len(result.trades), result.total_pnl_points))

    best = max(variants, key=lambda v: v[2])
    verdict = (
        "The COT bias filter did not improve TRAIN P&L, so it is reported and not "
        "adopted; adopting it would be a second pass of tuning on the same train data."
        if best[0] == variants[0][0]
        else (f"'{best[0]}' improved TRAIN P&L, but it is reported only: adopting it "
              f"after the parameters were already locked would be a second tuning pass "
              f"on the same train data.")
    )

    return {
        "available": True,
        "source": "CFTC legacy Commitments of Traders, futures-only (deacot history files)",
        "contract": COT_CONTRACT,
        "n_reports": len(weekly),
        "first_report": weekly.index[0].date(),
        "last_report": weekly.index[-1].date(),
        "lookback_weeks": int(cot_cfg.get("lookback_weeks", 156)),
        "variants": variants,
        "verdict": verdict,
    }
