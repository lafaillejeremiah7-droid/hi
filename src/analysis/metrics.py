"""Performance metrics calculation module.

Computes standard trading performance metrics from backtest results,
including scalping-specific metrics (trades per day, hold time,
session breakdown, consecutive winners/losers).
"""

from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestResult, Trade


def compute_sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """Compute annualized Sharpe ratio.

    Sharpe = (mean_return - risk_free) / std_return * sqrt(periods_per_year)

    Args:
        returns: Series of periodic returns.
        risk_free: Risk-free rate per period (default 0).
        periods_per_year: Number of trading periods per year.

    Returns:
        Annualized Sharpe ratio. Returns 0 if std is 0.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free
    return float(
        excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    )


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from equity curve.

    Max drawdown is the largest peak-to-trough decline.

    Args:
        equity_curve: Cumulative P&L or equity series.

    Returns:
        Maximum drawdown as a positive number (in same units as equity).
        Returns 0 if equity never declines.
    """
    if len(equity_curve) == 0:
        return 0.0

    running_max = equity_curve.cummax()
    drawdown = running_max - equity_curve
    return float(drawdown.max())


def compute_profit_factor(trades: list[Trade]) -> float:
    """Compute profit factor (gross profits / gross losses).

    Args:
        trades: List of completed trades.

    Returns:
        Profit factor. Returns inf if no losing trades, 0 if no winning trades.
    """
    if not trades:
        return 0.0

    gross_profit = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    gross_loss = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def compute_win_rate(trades: list[Trade]) -> float:
    """Compute win rate (fraction of profitable trades).

    Args:
        trades: List of completed trades.

    Returns:
        Win rate as a fraction (0-1). Returns 0 if no trades.
    """
    if not trades:
        return 0.0

    winners = sum(1 for t in trades if t.pnl_net > 0)
    return winners / len(trades)


def compute_avg_trade(trades: list[Trade]) -> float:
    """Compute average trade P&L (net of costs).

    Args:
        trades: List of completed trades.

    Returns:
        Average P&L per trade in points. Returns 0 if no trades.
    """
    if not trades:
        return 0.0

    return sum(t.pnl_net for t in trades) / len(trades)


def compute_total_return(equity_curve: pd.Series) -> float:
    """Compute total return from equity curve.

    Args:
        equity_curve: Cumulative P&L series.

    Returns:
        Total return (final equity value - initial value).
    """
    if len(equity_curve) == 0:
        return 0.0

    return float(equity_curve.iloc[-1] - equity_curve.iloc[0])


def compute_calmar_ratio(
    returns: pd.Series, max_dd: float, periods_per_year: int = 252
) -> float:
    """Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of periodic returns.
        max_dd: Maximum drawdown value (positive number).
        periods_per_year: Number of trading periods per year.

    Returns:
        Calmar ratio. Returns 0 if max drawdown is 0.
    """
    if max_dd == 0 or len(returns) == 0:
        return 0.0

    annualized_return = returns.mean() * periods_per_year
    return float(annualized_return / max_dd)


def compute_max_consecutive(trades: list[Trade]) -> tuple[int, int]:
    """Compute max consecutive winners and losers.

    Args:
        trades: List of completed trades.

    Returns:
        Tuple of (max_consecutive_winners, max_consecutive_losers).
    """
    if not trades:
        return 0, 0

    max_winners = 0
    max_losers = 0
    current_winners = 0
    current_losers = 0

    for t in trades:
        if t.pnl_net > 0:
            current_winners += 1
            current_losers = 0
            max_winners = max(max_winners, current_winners)
        elif t.pnl_net < 0:
            current_losers += 1
            current_winners = 0
            max_losers = max(max_losers, current_losers)
        else:
            current_winners = 0
            current_losers = 0

    return max_winners, max_losers


def compute_avg_hold_time_minutes(trades: list[Trade]) -> float:
    """Compute average hold time in minutes.

    Args:
        trades: List of completed trades with entry_time/exit_time.

    Returns:
        Average hold time in minutes. Returns 0 if cannot compute.
    """
    if not trades:
        return 0.0

    hold_times = []
    for t in trades:
        if t.entry_time is not None and t.exit_time is not None:
            try:
                delta = t.exit_time - t.entry_time
                minutes = delta.total_seconds() / 60
                hold_times.append(minutes)
            except (TypeError, AttributeError):
                continue

    if not hold_times:
        return 0.0

    return float(np.mean(hold_times))


def compute_trades_per_day(trades: list[Trade], total_trading_days: int | None = None) -> float:
    """Compute average number of trades per trading day.

    Args:
        trades: List of completed trades.
        total_trading_days: Total number of trading days in the dataset.
            If None, computed from trade dates.

    Returns:
        Average trades per day.
    """
    if not trades:
        return 0.0

    if total_trading_days and total_trading_days > 0:
        return len(trades) / total_trading_days

    # Compute from trade entry dates
    dates = set()
    for t in trades:
        if t.entry_time is not None and hasattr(t.entry_time, "date"):
            dates.add(t.entry_time.date())

    if not dates:
        return 0.0

    # Use the date range span, not just dates with trades
    min_date = min(dates)
    max_date = max(dates)
    # Approximate trading days as business days in the range
    if min_date == max_date:
        return float(len(trades))

    total_days = (max_date - min_date).days
    # Rough estimate: ~252 trading days per 365 calendar days
    approx_trading_days = max(1, int(total_days * 252 / 365))
    return len(trades) / approx_trading_days


def compute_session_breakdown(trades: list[Trade]) -> dict[str, dict[str, Any]]:
    """Compute win rate and metrics by trading session.

    Splits trades into AM session (before 12:00) and PM session (12:00+).

    Args:
        trades: List of completed trades.

    Returns:
        Dict with 'am' and 'pm' keys, each containing:
        {trades, wins, win_rate, avg_pnl}.
    """
    am_trades = []
    pm_trades = []

    for t in trades:
        if t.entry_time is not None and hasattr(t.entry_time, "hour"):
            if t.entry_time.hour < 12:
                am_trades.append(t)
            else:
                pm_trades.append(t)

    def session_stats(session_trades: list[Trade]) -> dict[str, Any]:
        if not session_trades:
            return {"trades": 0, "wins": 0, "win_rate": 0.0, "avg_pnl": 0.0}
        wins = sum(1 for t in session_trades if t.pnl_net > 0)
        return {
            "trades": len(session_trades),
            "wins": wins,
            "win_rate": wins / len(session_trades),
            "avg_pnl": sum(t.pnl_net for t in session_trades) / len(session_trades),
        }

    return {
        "am": session_stats(am_trades),
        "pm": session_stats(pm_trades),
    }


def compute_best_worst_hours(trades: list[Trade]) -> dict[str, Any]:
    """Compute best and worst trading hours by average P&L.

    Args:
        trades: List of completed trades.

    Returns:
        Dict with best_hour, worst_hour, and hourly_pnl breakdown.
    """
    hourly_pnl: dict[int, list[float]] = {}

    for t in trades:
        if t.entry_time is not None and hasattr(t.entry_time, "hour"):
            hour = t.entry_time.hour
            if hour not in hourly_pnl:
                hourly_pnl[hour] = []
            hourly_pnl[hour].append(t.pnl_net)

    if not hourly_pnl:
        return {"best_hour": None, "worst_hour": None, "hourly_avg": {}}

    hourly_avg = {h: np.mean(pnls) for h, pnls in hourly_pnl.items()}
    best_hour = max(hourly_avg, key=hourly_avg.get)
    worst_hour = min(hourly_avg, key=hourly_avg.get)

    return {
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "hourly_avg": hourly_avg,
    }


def compute_scalping_metrics(
    trades: list[Trade], total_trading_days: int | None = None, point_value: float = 20.0
) -> dict[str, Any]:
    """Compute scalping-specific metrics.

    Args:
        trades: List of completed trades.
        total_trading_days: Total trading days in dataset.
        point_value: Dollar value per point.

    Returns:
        Dict of scalping metrics.
    """
    max_winners, max_losers = compute_max_consecutive(trades)
    session_breakdown = compute_session_breakdown(trades)
    hour_analysis = compute_best_worst_hours(trades)

    avg_hold_minutes = compute_avg_hold_time_minutes(trades)
    trades_per_day = compute_trades_per_day(trades, total_trading_days)

    # EV per trade in dollars
    if trades:
        avg_pnl_pts = sum(t.pnl_net for t in trades) / len(trades)
        ev_per_trade_dollars = avg_pnl_pts * point_value
    else:
        ev_per_trade_dollars = 0.0

    return {
        "avg_trades_per_day": trades_per_day,
        "avg_hold_time_minutes": avg_hold_minutes,
        "max_consecutive_winners": max_winners,
        "max_consecutive_losers": max_losers,
        "ev_per_trade_dollars": ev_per_trade_dollars,
        "session_breakdown": session_breakdown,
        "best_hour": hour_analysis.get("best_hour"),
        "worst_hour": hour_analysis.get("worst_hour"),
    }


def compute_all_metrics(result: BacktestResult, point_value: float = 20.0) -> dict[str, Any]:
    """Compute all performance metrics for a backtest result.

    Args:
        result: BacktestResult from the engine.
        point_value: Dollar value per point.

    Returns:
        Dictionary with all computed metrics.
    """
    trades = result.trades
    equity = result.equity_net

    # Compute periodic returns from equity curve
    if len(equity) > 1:
        returns = equity.diff().fillna(0)
    else:
        returns = pd.Series(dtype=float)

    max_dd = compute_max_drawdown(equity)

    metrics = {
        "total_trades": len(trades),
        "total_return": compute_total_return(equity),
        "sharpe_ratio": compute_sharpe_ratio(returns),
        "max_drawdown": max_dd,
        "profit_factor": compute_profit_factor(trades),
        "win_rate": compute_win_rate(trades),
        "avg_trade": compute_avg_trade(trades),
        "calmar_ratio": compute_calmar_ratio(returns, max_dd),
    }

    # Additional metrics
    if trades:
        metrics["best_trade"] = max(t.pnl_net for t in trades)
        metrics["worst_trade"] = min(t.pnl_net for t in trades)
        metrics["avg_winner"] = (
            np.mean([t.pnl_net for t in trades if t.pnl_net > 0])
            if any(t.pnl_net > 0 for t in trades)
            else 0.0
        )
        metrics["avg_loser"] = (
            np.mean([t.pnl_net for t in trades if t.pnl_net < 0])
            if any(t.pnl_net < 0 for t in trades)
            else 0.0
        )
        metrics["total_costs"] = sum(t.cost for t in trades)
    else:
        metrics["best_trade"] = 0.0
        metrics["worst_trade"] = 0.0
        metrics["avg_winner"] = 0.0
        metrics["avg_loser"] = 0.0
        metrics["total_costs"] = 0.0

    # Expected Value (EV) per trade = (avg_win * win_rate) - (avg_loss * loss_rate)
    win_rate = metrics["win_rate"]
    loss_rate = 1.0 - win_rate
    avg_winner = metrics["avg_winner"]
    avg_loser = abs(metrics["avg_loser"])
    metrics["expected_value"] = (avg_winner * win_rate) - (avg_loser * loss_rate)

    # Scalping metrics
    scalping = compute_scalping_metrics(trades, point_value=point_value)
    metrics["avg_trades_per_day"] = scalping["avg_trades_per_day"]
    metrics["avg_hold_time_minutes"] = scalping["avg_hold_time_minutes"]
    metrics["max_consecutive_winners"] = scalping["max_consecutive_winners"]
    metrics["max_consecutive_losers"] = scalping["max_consecutive_losers"]
    metrics["ev_per_trade_dollars"] = scalping["ev_per_trade_dollars"]

    return metrics
