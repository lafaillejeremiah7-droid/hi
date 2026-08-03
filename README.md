# NAS100 Backtesting Framework

A comprehensive Python backtesting framework for NASDAQ 100 futures (NQ) implementing two distinct trading strategies with proper backtesting methodology: realistic cost modeling, train/validation split for overfitting prevention, and Monte Carlo simulation for robustness assessment.

## Features

- **Two trading strategies**: Order Flow and Volume Profile
- **Realistic cost modeling**: Volatility-based slippage (0.5-2 points) and configurable commissions
- **Train/validation split**: Parameter optimization on training data only, validation with fixed parameters
- **Monte Carlo simulation**: 10,000 randomized trade sequences for robustness testing
- **Interactive HTML reports**: Plotly-based visualizations with equity curves, Monte Carlo paths, and metrics tables
- **Configurable via YAML**: All strategy parameters, costs, and simulation settings in one config file

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd nas100-backtest

# Install dependencies
uv sync
```

### Run the Backtest

```bash
uv run python -m src.main
```

This will:
1. Download 5 years of NAS100 data (via yfinance)
2. Preprocess data with order flow and volume profile indicators
3. Run both strategies with train/validation split (60%/40%)
4. Perform Monte Carlo simulation (10,000 paths)
5. Generate HTML reports in `results/`
6. Save trade logs as CSV files

### Run Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
.
├── config/
│   └── default.yaml          # All configurable parameters
├── data/                     # Cached market data (auto-created, gitignored)
├── results/                  # Generated reports and trade logs (gitignored)
│   ├── order_flow_report.html
│   ├── order_flow_trades.csv
│   ├── volume_profile_report.html
│   └── volume_profile_trades.csv
├── src/
│   ├── analysis/
│   │   ├── metrics.py        # Performance metrics (Sharpe, drawdown, etc.)
│   │   └── monte_carlo.py    # Monte Carlo simulation engine
│   ├── backtester/
│   │   ├── costs.py          # Slippage and commission modeling
│   │   └── engine.py         # Core backtesting engine with train/val split
│   ├── data/
│   │   ├── fetcher.py        # Market data download (yfinance)
│   │   └── preprocessor.py   # Order flow proxy calculation
│   ├── indicators/
│   │   ├── order_flow.py     # Order flow indicator functions
│   │   └── volume_profile.py # Volume profile indicator functions
│   ├── reports/
│   │   └── generator.py      # HTML report generation (Plotly)
│   ├── strategies/
│   │   ├── base.py           # Abstract base strategy class
│   │   ├── order_flow_strategy.py
│   │   └── volume_profile_strategy.py
│   ├── config.py             # Configuration loader
│   └── main.py               # Pipeline orchestrator
├── tests/
│   ├── test_backtester.py
│   ├── test_indicators.py
│   ├── test_monte_carlo.py
│   └── test_strategies.py
├── pyproject.toml
└── README.md
```

## Configuration Guide

All parameters are in `config/default.yaml`:

### Data Section

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | `NQ=F` | Primary yfinance symbol for NQ futures |
| `fallback_symbol` | `^NDX` | Fallback if primary fails |
| `period` | `5y` | Data history length |
| `interval` | `1d` | Bar interval (daily) |
| `train_split` | `0.6` | Fraction of data for training |
| `cache_dir` | `data` | Directory for cached data |

### Costs Section

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slippage_points` | `1.0` | Base slippage per trade (scales with volatility) |
| `commission_per_round_trip` | `4.50` | Commission in dollars per round trip |
| `point_value` | `20.0` | Dollar value per point (NQ micro = $20) |

### Monte Carlo Section

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulations` | `10000` | Number of simulation paths |
| `confidence_level` | `0.95` | Confidence level for intervals |
| `ruin_threshold` | `0.50` | Fraction of equity loss defining ruin |
| `seed` | `42` | Random seed for reproducibility |

### Strategy Sections

See `config/default.yaml` for the full list of `order_flow_strategy` and `volume_profile_strategy` parameters including thresholds, lookback periods, and ATR multipliers.

## Strategy Descriptions

### Order Flow Strategy

Combines five order flow proxy signals derived from OHLCV data:

1. **Absorption**: Detects high volume at support/resistance zones where buying/selling pressure is being absorbed, signaling potential reversals.
2. **Cumulative Delta Divergence**: Identifies divergences between price direction and cumulative buying/selling pressure.
3. **Stacked Imbalances**: Finds consecutive bars where one side of volume is 3x more aggressive, creating strong S/R zones.
4. **Failed Auctions**: Detects candles with improper auction completion, identifying price levels likely to be revisited.
5. **Trapped Traders**: Spots heavy volume at candle extremes followed by reversals, catching traders on the wrong side.

Signals are only valid when they occur at pre-identified support/resistance zones. A minimum of 2 sub-signals must align to generate a trade.

### Volume Profile Strategy

Three setups based on volume distribution analysis:

1. **Volume Accumulation**: Finds consolidation zones with heavy volume, then enters on pullback to the high-volume node after a breakout.
2. **Trend Setup**: During strong trends, identifies volume clusters where institutions added positions and enters on pullback to those levels.
3. **Support/Resistance Flip**: Trades when previously heavy support zones become resistance (or vice versa).

Stop losses are placed in low-volume areas (where price moves fast), and take profits at the next heavy-volume zone (natural barriers).

## Methodology Notes

### OHLCV Proxy Approach

Since true order flow data (Level 2, time and sales) is not freely available for 5+ years, this framework uses OHLCV-based proxies:

- **Volume Delta Proxy**: If close > open, more volume is attributed to the ask (buying); if close < open, to the bid (selling).
- **Absorption Proxy**: High relative volume at known S/R levels serves as an absorption signal.
- **Imbalance Proxy**: Relative volume comparisons between adjacent bars model order flow imbalances.
- **Volume Profile**: Built from price-weighted volume distribution across bars in a lookback window.

These proxies are less precise than true order flow data but capture the same underlying dynamics when applied to daily bars over a 5-year period.

### Backtesting Methodology

The framework follows four key rules for proper backtesting:

1. **Realistic Costs**: Volatility-scaled slippage (higher volatility = worse fills) plus fixed commissions. Equity curves shown both with and without costs.
2. **Long Time Period**: 5 years of data ensures strategies are tested across multiple market regimes.
3. **Train/Validation Split**: Parameters are optimized only on the first 60% of data. The last 40% tests with locked parameters to detect overfitting.
4. **Monte Carlo Simulation**: Randomizes trade sequences to assess robustness beyond the specific historical ordering.

## Output

After running the pipeline, the `results/` directory contains:

- `order_flow_report.html` - Full interactive report for Order Flow strategy
- `order_flow_trades.csv` - Individual trade log with entry/exit times and P&L
- `volume_profile_report.html` - Full interactive report for Volume Profile strategy
- `volume_profile_trades.csv` - Individual trade log

Each HTML report includes:
- Equity curves (gross vs net, overlaid)
- Train vs validation performance side-by-side
- Monte Carlo simulation paths with confidence bands
- Final equity distribution histogram
- Complete metrics tables for both periods
- Optimized parameters from training

## License

MIT
