"""HTML report generation module.

Generates self-contained HTML reports with interactive Plotly charts
for backtest results, Monte Carlo simulations, and performance metrics.
Includes scalping-specific metrics and micro-validation sections.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.monte_carlo import MonteCarloResults
from src.backtester.engine import BacktestResult


def plot_equity_curves(result: BacktestResult, point_value: float = 20.0) -> go.Figure:
    """Create overlaid equity curves (gross vs net).

    Shows equity curves both with and without costs for comparison.

    Args:
        result: BacktestResult with equity_gross and equity_net.
        point_value: Dollar value per point.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Convert from points to dollars
    equity_gross_dollars = result.equity_gross * point_value
    equity_net_dollars = result.equity_net * point_value

    x_values = result.equity_gross.index

    fig.add_trace(go.Scatter(
        x=x_values,
        y=equity_gross_dollars,
        mode="lines",
        name="Equity (Gross - No Costs)",
        line=dict(color="blue", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=x_values,
        y=equity_net_dollars,
        mode="lines",
        name="Equity (Net - With Costs)",
        line=dict(color="red", width=2),
    ))

    fig.update_layout(
        title="Equity Curves: Gross vs Net",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450,
    )

    return fig


def plot_train_vs_validation(
    train_result: BacktestResult | None,
    val_result: BacktestResult | None,
    point_value: float = 20.0,
) -> go.Figure:
    """Create train vs validation equity comparison.

    Args:
        train_result: BacktestResult from training period.
        val_result: BacktestResult from validation period.
        point_value: Dollar value per point.

    Returns:
        Plotly Figure with side-by-side comparison.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Period", "Validation Period"),
        horizontal_spacing=0.08,
    )

    if train_result is not None and len(train_result.equity_net) > 0:
        train_eq = train_result.equity_net * point_value
        fig.add_trace(go.Scatter(
            x=train_eq.index,
            y=train_eq.values,
            mode="lines",
            name="Train (Net)",
            line=dict(color="green", width=2),
        ), row=1, col=1)

        train_eq_gross = train_result.equity_gross * point_value
        fig.add_trace(go.Scatter(
            x=train_eq_gross.index,
            y=train_eq_gross.values,
            mode="lines",
            name="Train (Gross)",
            line=dict(color="lightgreen", width=1, dash="dash"),
        ), row=1, col=1)

    if val_result is not None and len(val_result.equity_net) > 0:
        val_eq = val_result.equity_net * point_value
        fig.add_trace(go.Scatter(
            x=val_eq.index,
            y=val_eq.values,
            mode="lines",
            name="Validation (Net)",
            line=dict(color="purple", width=2),
        ), row=1, col=2)

        val_eq_gross = val_result.equity_gross * point_value
        fig.add_trace(go.Scatter(
            x=val_eq_gross.index,
            y=val_eq_gross.values,
            mode="lines",
            name="Validation (Gross)",
            line=dict(color="plum", width=1, dash="dash"),
        ), row=1, col=2)

    fig.update_layout(
        title="Train vs Validation Performance",
        template="plotly_white",
        height=400,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=2)

    return fig


def plot_monte_carlo(mc_results: MonteCarloResults, max_paths_shown: int = 200) -> go.Figure:
    """Create Monte Carlo visualization.

    Includes spaghetti plot of sample paths with median highlighted,
    confidence bands, and a histogram of final equity distribution.

    Args:
        mc_results: MonteCarloResults from simulation.
        max_paths_shown: Maximum number of paths to show (for performance).

    Returns:
        Plotly Figure with Monte Carlo visualization.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Monte Carlo Equity Paths", "Final Equity Distribution"),
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.08,
    )

    paths = mc_results.paths
    n_sims, n_steps = paths.shape
    x_axis = np.arange(n_steps)

    # Show a subset of paths for visual clarity
    n_show = min(max_paths_shown, n_sims)
    step = max(1, n_sims // n_show)
    sample_indices = np.arange(0, n_sims, step)[:n_show]

    # Individual paths (semi-transparent)
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=paths[idx],
            mode="lines",
            line=dict(color="rgba(100,149,237,0.05)", width=0.5),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

    # Median path
    median_path = np.median(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=median_path,
        mode="lines",
        name="Median",
        line=dict(color="blue", width=3),
    ), row=1, col=1)

    # Confidence bands
    lower_band = np.percentile(paths, 5, axis=0)
    upper_band = np.percentile(paths, 95, axis=0)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=upper_band,
        mode="lines",
        name="95th Percentile",
        line=dict(color="green", width=1.5, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=lower_band,
        mode="lines",
        name="5th Percentile",
        line=dict(color="red", width=1.5, dash="dash"),
    ), row=1, col=1)

    # Starting capital reference line
    fig.add_trace(go.Scatter(
        x=[0, n_steps - 1],
        y=[paths[0, 0], paths[0, 0]],
        mode="lines",
        name="Starting Capital",
        line=dict(color="gray", width=1, dash="dot"),
    ), row=1, col=1)

    # Histogram of final equities
    fig.add_trace(go.Histogram(
        x=mc_results.final_equities,
        nbinsx=50,
        name="Final Equity",
        marker_color="rgba(100,149,237,0.7)",
        showlegend=False,
    ), row=1, col=2)

    # Add vertical lines for median and percentiles
    fig.add_vline(
        x=mc_results.median_final_equity,
        line_dash="solid",
        line_color="blue",
        annotation_text="Median",
        row=1, col=2,
    )
    fig.add_vline(
        x=mc_results.confidence_interval_lower,
        line_dash="dash",
        line_color="red",
        annotation_text="5th %ile",
        row=1, col=2,
    )
    fig.add_vline(
        x=mc_results.confidence_interval_upper,
        line_dash="dash",
        line_color="green",
        annotation_text="95th %ile",
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Monte Carlo Simulation ({mc_results.n_simulations:,} paths, {mc_results.n_trades_per_sim} trades each)",
        template="plotly_white",
        height=450,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Trade #", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_xaxes(title_text="Final Equity ($)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return fig


def create_metrics_table(
    metrics: dict[str, Any],
    title: str = "Performance Metrics",
) -> str:
    """Create an HTML table for performance metrics.

    Args:
        metrics: Dictionary of metric name -> value.
        title: Table title.

    Returns:
        HTML string for the metrics table.
    """
    rows = ""
    for key, value in metrics.items():
        display_name = key.replace("_", " ").title()
        if isinstance(value, float):
            if "rate" in key:
                formatted = f"{value:.2%}"
            elif "ratio" in key or "factor" in key:
                formatted = f"{value:.3f}"
            elif "drawdown" in key or "return" in key or "trade" in key or "cost" in key or "value" in key or "dollars" in key:
                formatted = f"${value:.2f}" if abs(value) < 1e8 else f"{value:.2f}"
            elif "minutes" in key:
                formatted = f"{value:.1f} min"
            elif "per_day" in key:
                formatted = f"{value:.2f}"
            else:
                formatted = f"{value:.2f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

        rows += f"<tr><td><strong>{display_name}</strong></td><td>{formatted}</td></tr>\n"

    return f"""
    <div class="metrics-table">
        <h3>{title}</h3>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def create_monte_carlo_summary_table(mc_results: MonteCarloResults) -> str:
    """Create HTML summary table for Monte Carlo results.

    Args:
        mc_results: MonteCarloResults from simulation.

    Returns:
        HTML string with Monte Carlo summary.
    """
    rows = f"""
    <tr><td><strong>Simulations</strong></td><td>{mc_results.n_simulations:,}</td></tr>
    <tr><td><strong>Trades Per Simulation</strong></td><td>{mc_results.n_trades_per_sim:,}</td></tr>
    <tr><td><strong>Median Final Equity</strong></td><td>${mc_results.median_final_equity:,.2f}</td></tr>
    <tr><td><strong>Mean Final Equity</strong></td><td>${mc_results.mean_final_equity:,.2f}</td></tr>
    <tr><td><strong>95% CI Lower</strong></td><td>${mc_results.confidence_interval_lower:,.2f}</td></tr>
    <tr><td><strong>95% CI Upper</strong></td><td>${mc_results.confidence_interval_upper:,.2f}</td></tr>
    <tr><td><strong>Worst 5% Max Drawdown</strong></td><td>${mc_results.worst_5pct_drawdown:,.2f}</td></tr>
    <tr><td><strong>Probability of Ruin</strong></td><td>{mc_results.probability_of_ruin:.2%}</td></tr>
    """

    return f"""
    <div class="metrics-table">
        <h3>Monte Carlo Simulation Summary</h3>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def _create_walk_forward_chart(wf_results) -> go.Figure:
    """Create bar chart of per-window OOS returns.

    Args:
        wf_results: WalkForwardResults object.

    Returns:
        Plotly Figure with bar chart.
    """
    fig = go.Figure()

    labels = []
    values = []
    colors = []

    for w in wf_results.windows:
        start_str = str(w.test_start)[:10] if hasattr(w.test_start, 'strftime') else str(w.test_start)[:10]
        labels.append(f"W{w.window_idx + 1}\n{start_str}")
        ret = w.test_metrics.get("total_return", 0.0)
        values.append(ret)
        colors.append("rgba(46, 204, 113, 0.7)" if ret > 0 else "rgba(231, 76, 60, 0.7)")

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        name="OOS Return (pts)",
    ))

    fig.update_layout(
        title=f"Walk-Forward: Per-Window OOS Returns ({wf_results.consistency_ratio:.0%} profitable)",
        xaxis_title="Window",
        yaxis_title="Return (points)",
        template="plotly_white",
        height=350,
    )

    return fig


def _create_walk_forward_equity(wf_results) -> go.Figure:
    """Create walk-forward stitched OOS equity curve.

    Args:
        wf_results: WalkForwardResults object.

    Returns:
        Plotly Figure with equity curve.
    """
    fig = go.Figure()

    eq = wf_results.combined_oos_equity
    if len(eq) > 0:
        fig.add_trace(go.Scatter(
            x=eq.index,
            y=eq.values,
            mode="lines",
            name="Combined OOS Equity",
            line=dict(color="purple", width=2),
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Walk-Forward: Stitched OOS Equity Curve",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (points)",
        template="plotly_white",
        height=350,
    )

    return fig


def _create_oos_equity_chart(oos_result: BacktestResult, point_value: float) -> go.Figure:
    """Create equity curve chart for True OOS period.

    Args:
        oos_result: BacktestResult from OOS period.
        point_value: Dollar value per point.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    eq_dollars = oos_result.equity_net * point_value

    fig.add_trace(go.Scatter(
        x=eq_dollars.index,
        y=eq_dollars.values,
        mode="lines",
        name="True OOS (Net)",
        line=dict(color="darkred", width=2),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="True Out-of-Sample Equity (NEVER touched during optimization)",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        template="plotly_white",
        height=400,
    )

    return fig


def generate_full_report(
    strategy_name: str,
    backtest_result: BacktestResult,
    mc_results: MonteCarloResults,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    oos_metrics: dict[str, Any] | None = None,
    point_value: float = 20.0,
    output_path: str | None = None,
    walk_forward_results: Any = None,
    micro_result: Any = None,
) -> str:
    """Generate a complete self-contained HTML report.

    Assembles equity curves, train vs validation vs OOS comparison, Monte Carlo
    visualization, walk-forward analysis, and metrics tables into a single HTML file.
    Includes scalping-specific metrics section and final verdict.

    Args:
        strategy_name: Human-readable strategy name.
        backtest_result: Full BacktestResult with train/val/oos sub-results.
        mc_results: MonteCarloResults from simulation.
        train_metrics: Metrics dict for training period.
        validation_metrics: Metrics dict for validation period.
        oos_metrics: Metrics dict for True OOS period (optional).
        point_value: Dollar value per point.
        output_path: Path to save the HTML file. If None, returns HTML string.
        walk_forward_results: WalkForwardResults (optional).
        micro_result: BacktestResult from micro-validation (optional).

    Returns:
        HTML string of the complete report.
    """
    if oos_metrics is None:
        oos_metrics = {}

    # Generate plotly chart divs
    equity_fig = plot_equity_curves(backtest_result, point_value)
    train_val_fig = plot_train_vs_validation(
        backtest_result.train_result,
        backtest_result.validation_result,
        point_value,
    )
    mc_fig = plot_monte_carlo(mc_results)

    # Convert figures to HTML divs
    equity_html = equity_fig.to_html(full_html=False, include_plotlyjs=False)
    train_val_html = train_val_fig.to_html(full_html=False, include_plotlyjs=False)
    mc_html = mc_fig.to_html(full_html=False, include_plotlyjs=False)

    # Create metrics tables (filter out non-displayable items)
    display_train = {k: v for k, v in train_metrics.items() if not isinstance(v, (dict, list))}
    display_val = {k: v for k, v in validation_metrics.items() if not isinstance(v, (dict, list))}
    display_oos = {k: v for k, v in oos_metrics.items() if not isinstance(v, (dict, list))}

    train_table = create_metrics_table(display_train, "Training Period Metrics")
    val_table = create_metrics_table(display_val, "Validation Period Metrics") if display_val else ""
    oos_table = create_metrics_table(display_oos, "True OOS Period Metrics") if display_oos else ""
    mc_summary = create_monte_carlo_summary_table(mc_results)

    # Best parameters section
    params_html = ""
    if backtest_result.best_params:
        params_rows = ""
        for k, v in backtest_result.best_params.items():
            params_rows += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>\n"
        params_html = f"""
        <div class="metrics-table">
            <h3>Optimized Parameters (from training)</h3>
            <table>
                <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                <tbody>{params_rows}</tbody>
            </table>
        </div>
        """

    # Scalping metrics section
    scalping_html = ""
    scalping_keys = [
        "avg_trades_per_day", "avg_hold_time_minutes",
        "max_consecutive_winners", "max_consecutive_losers",
        "ev_per_trade_dollars",
    ]
    scalping_metrics = {k: v for k, v in train_metrics.items() if k in scalping_keys}
    if scalping_metrics:
        scalping_html = create_metrics_table(scalping_metrics, "Scalping Metrics")

    # Three-Way Split section
    three_way_html = ""
    if display_oos:
        three_way_html = f"""
    <div class="section">
        <h2>Three-Way Split: Train vs Validation vs True OOS</h2>
        <p><em>True OOS data was NEVER seen during optimization or parameter selection.
        This is the final, unbiased evaluation.</em></p>
        <div class="metrics-container" style="grid-template-columns: 1fr 1fr 1fr;">
            {train_table}
            {val_table}
            {oos_table}
        </div>
    </div>
        """

    # Walk-forward section
    walk_forward_html = ""
    if walk_forward_results is not None and walk_forward_results.total_oos_windows > 0:
        wf = walk_forward_results
        # Per-window returns bar chart
        wf_fig = _create_walk_forward_chart(wf)
        wf_chart_html = wf_fig.to_html(full_html=False, include_plotlyjs=False)

        # Equity curve
        wf_eq_fig = _create_walk_forward_equity(wf)
        wf_eq_html = wf_eq_fig.to_html(full_html=False, include_plotlyjs=False)

        # Per-window metrics table
        wf_table_rows = ""
        for w in wf.windows:
            test_return = w.test_metrics.get("total_return", 0.0)
            test_trades = w.test_metrics.get("total_trades", 0)
            test_wr = w.test_metrics.get("win_rate", 0.0)
            test_sharpe = w.test_metrics.get("sharpe_ratio", 0.0)
            test_ev = w.test_metrics.get("expected_value", 0.0)
            start_str = str(w.test_start)[:10] if hasattr(w.test_start, 'strftime') else str(w.test_start)[:10]
            end_str = str(w.test_end)[:10] if hasattr(w.test_end, 'strftime') else str(w.test_end)[:10]
            status = "+" if test_return > 0 else "-"
            wf_table_rows += (
                f"<tr><td>{w.window_idx + 1}</td>"
                f"<td>{start_str}</td><td>{end_str}</td>"
                f"<td>{test_trades}</td>"
                f"<td>{test_wr:.1%}</td>"
                f"<td>{test_ev:.2f}</td>"
                f"<td>{test_sharpe:.3f}</td>"
                f"<td>{test_return:.2f} pts {status}</td></tr>\n"
            )

        walk_forward_html = f"""
    <div class="section">
        <h2>Walk-Forward Analysis</h2>
        <div class="summary-box">
            <p><strong>Consistency Ratio:</strong> {wf.consistency_ratio:.0%}
            ({wf.profitable_windows}/{wf.total_oos_windows} months profitable) |
            <strong>Walk-Forward Efficiency:</strong> {wf.walk_forward_efficiency:.3f} |
            <strong>Degradation:</strong> {wf.degradation_metric:.3f}</p>
            {"<p style='color: #ff6b6b;'><strong>WARNING:</strong> Performance degrading over time (regime sensitivity detected)</p>" if wf.regime_warning else ""}
        </div>
        <h3>Walk-Forward OOS Equity Curve (Stitched)</h3>
        {wf_eq_html}
        <h3>Per-Month OOS Returns</h3>
        {wf_chart_html}
        <h3>Per-Window Metrics</h3>
        <table>
            <thead><tr><th>#</th><th>Test Start</th><th>Test End</th>
            <th>Trades</th><th>Win Rate</th><th>EV (pts)</th>
            <th>Sharpe</th><th>Return</th></tr></thead>
            <tbody>{wf_table_rows}</tbody>
        </table>
    </div>
        """

    # OOS section in equity comparison
    oos_equity_html = ""
    if backtest_result.oos_result and len(getattr(backtest_result.oos_result, 'equity_net', pd.Series())) > 0:
        oos_fig = _create_oos_equity_chart(backtest_result.oos_result, point_value)
        oos_equity_html = f"""
    <div class="section">
        <h2>True OOS Equity Curve (Untouched Data)</h2>
        {oos_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
        """

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{strategy_name} - Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #16213e;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #16213e;
            margin-top: 40px;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .metrics-table {{
            margin: 10px 0;
        }}
        .metrics-table h3 {{
            color: #16213e;
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #16213e;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-box h2 {{
            color: white;
            margin-top: 0;
        }}
        .verdict-pass {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .verdict-fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            text-align: right;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <h1>{strategy_name} - Backtest Report</h1>

    <div class="summary-box">
        <h2>Strategy Summary</h2>
        <p><strong>Total Trades:</strong> {len(backtest_result.trades)} |
           <strong>Train Trades:</strong> {len(backtest_result.train_result.trades) if backtest_result.train_result else 0} |
           <strong>Validation Trades:</strong> {len(backtest_result.validation_result.trades) if backtest_result.validation_result else 0} |
           <strong>True OOS Trades:</strong> {len(backtest_result.oos_result.trades) if backtest_result.oos_result else 0}</p>
        <p><strong>Net P&L (Full Period):</strong> ${(backtest_result.equity_net.iloc[-1] if len(backtest_result.equity_net) > 0 else 0) * point_value:,.2f}</p>
    </div>

    <div class="section">
        <h2>Equity Curves</h2>
        {equity_html}
    </div>

    <div class="section">
        <h2>Train vs Validation</h2>
        {train_val_html}
    </div>

    {oos_equity_html}

    {three_way_html if three_way_html else f'''
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics-container">
            {train_table}
            {val_table}
        </div>
        {params_html}
    </div>
    '''}

    {walk_forward_html}

    <div class="section">
        <h2>Scalping Metrics</h2>
        {scalping_html}
    </div>

    <div class="section">
        <h2>Monte Carlo Simulation</h2>
        {mc_html}
        {mc_summary}
    </div>

    <p class="timestamp">Generated by NAS100 Backtesting Framework (Multi-Layer Anti-Overfitting)</p>
</body>
</html>"""

    # Save to file if path provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html
