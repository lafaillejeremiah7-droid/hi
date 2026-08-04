# NAS100 Backtesting Framework

Backtesting framework for NASDAQ 100 futures (NQ) on 5-minute bars with **real
order flow** (bid/ask volume and delta aggregated from Databento trades).

The pipeline runs one deliberately simple strategy and validates it honestly:
interleaved-block train/validation/OOS split, look-ahead regression tests,
walk-forward analysis, Monte Carlo, and a FundedNext 50K prop-firm sizing
analysis that respects the contract cap and the daily loss limit.

## Quick Start

```bash
uv sync
uv run python -m src.main        # full pipeline
uv run pytest tests/ -q          # test suite
```

Data: `data/NQ_5min_real_orderflow.parquet` (264,057 bars, Jan 2021 - Sep 2024).
Columns: `open, high, low, close, volume, bid_volume, ask_volume, delta,
trade_count, avg_trade_size`, where `delta = bid_volume - ask_volume` so a
**positive delta means net buying**.

## The Strategy

`SimpleStrategy` (`src/strategies/simple_strategy.py`) has exactly two entry
conditions - the core idea shared by the order flow and volume profile
methodologies:

1. **Level** - price is within `level_proximity_points` of the heavy-volume
   node: the price bin that traded the most volume over the last
   `profile_lookback` bars. That is where institutions positioned.
2. **Confirmation** - the rolling delta Z-score at that level says which side
   is winning. Positive delta while price sits at or above the node (support)
   is a long; negative delta at or below the node (resistance) is a short.

Exits are fixed points only: `stop_points` and `target_points`. No partial
closes, no trailing stop, no staged stop advancement, no time-based exit.

The entire tunable surface:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `profile_lookback` | 78 | Bars in the volume profile window (one 6.5h session) |
| `level_proximity_points` | 10 | How close price must be to the node |
| `delta_threshold` | 1.0 | Minimum absolute delta Z-score |
| `stop_points` | 20 | Fixed stop distance |
| `target_points` | 30 | Fixed target distance |
| `max_trades_per_day` | 2 | Daily entry cap (enforced by the engine) |
| `trading_session_start` / `_end` | 09:30 / 16:00 ET | Session filter |

Two constants are fixed by the instrument rather than tuned: the profile bin
width (5 points) and the delta Z-score window (50 bars).

`OrderFlowStrategy` and `VolumeProfileStrategy` are kept only as **baselines**
for the OOS comparison table the pipeline prints.

## Validation Design

### Interleaved-block split

The timeline is cut into consecutive 1-month blocks assigned round-robin:
`train, train, validation, oos`. Each split therefore spans 2021 through 2024
instead of one contiguous era (~50% / 25% / 25% of bars). Every block's exact
date range and bar count is printed so the split is auditable.

Each block is simulated on its own, so **no trade straddles a block
boundary**: a position still open on a block's last bar is closed at that
bar's close (`close_at_end`) and recorded in that block's split.

The pipeline prints a warning about a real artifact of this scheme: a 4-long
assignment cycle divides evenly into the 12-month year, so each split always
receives the same calendar months (train = Jan/Feb/May/Jun/Sep/Oct,
validation = Mar/Jul/Nov, OOS = Apr/Aug/Dec). Month-of-year seasonality is
perfectly confounded with the split.

### Parameter selection

The only tuning allowed: score the 4x5x3x2 = 120 grid combinations on **train
only**, promote the top 5 to **validation**, lock the single best, then touch
**OOS once**. Train, validation and OOS are printed side by side so
degradation is visible.

### Look-ahead guards

`tests/test_lookahead.py` asserts that the signal at bar *i* is unchanged when
every bar after *i* is deleted (and separately when it is replaced with
noise), for the strategy, the volume profile level, and the delta Z-score.

## Prop Firm Analysis (FundedNext 50K)

Rules modelled: $50,000 account, $3,000 profit target, $1,000 daily loss
limit, $48,000 equity floor, up to 40 Micro ($2/pt) or 4 Mini ($20/pt).

1. **Hard filter** - one stop-out costs `stop_points x $/pt x contracts`. Any
   size where a single stop-out exceeds the $1,000 daily limit is eliminated
   before any simulation, and the eliminated range is stated explicitly.
2. **Monte Carlo** - for every surviving size, 10,000 simulated 250-day years.
   Trades are resampled from the OOS trade list and replayed one at a time;
   the daily loss limit and the equity floor are **hard fails checked inside
   the loop**. A breached account is dead and keeps the equity it had at that
   instant. Nothing is capped after the fact.
3. **Arithmetic ceiling** - what 40 Micro, the observed trades/day and the OOS
   EV/trade would produce with zero losing days and no compounding. This is
   the hard upper bound the contract cap imposes, and it is compared directly
   against the 11,376% figure this project was previously asked to justify.

## Project Structure

```
config/default.yaml              All parameters, grid and prop firm rules
src/
  strategies/
    simple_strategy.py           PRIMARY strategy (level + delta)
    order_flow_strategy.py       baseline
    volume_profile_strategy.py   baseline
    base.py
  backtester/
    engine.py                    Bar-by-bar simulation, session + daily caps
    block_split.py               Interleaved-block split and per-block runs
    costs.py                     Volatility-scaled slippage + commission
  analysis/
    parameter_selection.py       Train -> validation -> lock
    prop_firm.py                 FN 50K sizing, MC, arithmetic ceiling
    walk_forward.py              Rolling walk-forward windows
    metrics.py, monte_carlo.py
  data/                          Fetching, trade aggregation, preprocessing
  indicators/                    order_flow.py, volume_profile.py
  reports/generator.py           Plotly HTML reports
  main.py                        Pipeline orchestrator
tests/                           Unit, look-ahead and split regression tests
```

## Known Caveats

These are printed by the pipeline as well:

- **Overnight holds.** Bars outside 09:30-16:00 ET are filtered out before the
  strategy runs, so a position held past the close never has its stop tested
  against the overnight tape. The pipeline reports how many trades and how
  much P&L this affects.
- **Month confounding.** See the split section above.
- **The contract cap dominates the return question.** Because the maximum size
  does not grow with equity, returns accumulate linearly, not exponentially.

## License

MIT
