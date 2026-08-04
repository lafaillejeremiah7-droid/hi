# NAS100 Backtesting Framework

Backtesting framework for NASDAQ 100 futures (NQ). Two independent studies run
end to end:

1. **A Larry Williams daily-bar system** (primary) - Greatest Swing Value,
   Oops!, Smash Day and TDOM seasonality on daily RTH bars, with stop-order
   entries, multi-day holds and explicit overnight gap modelling.
2. **`SimpleStrategy` on 5-minute bars with real order flow** (baseline) - kept
   unchanged for comparison.

Both are validated the same way: a train/validation/OOS split that is printed
and auditable, a capped parameter grid scored on train only, look-ahead
regression tests, walk-forward, and a FundedNext 50K sizing analysis that
respects the contract cap and the daily loss limit.

## Quick Start

```bash
uv sync
uv run python -m src.main        # both studies
uv run pytest tests/ -q          # test suite
```

Data: `data/NQ_5min_real_orderflow.parquet` (264,057 bars, Jan 2021 - Sep 2024).
Columns: `open, high, low, close, volume, bid_volume, ask_volume, delta,
trade_count, avg_trade_size`, where `delta = bid_volume - ask_volume` so a
**positive delta means net buying**.

## The Williams Daily System (primary)

Daily RTH bars (09:30-16:00 ET) are resampled from
`data/NQ_1min_clean_2021_2026.parquet` and cached to `data/NQ_daily_rth.parquet`
(1,417 trading days, Jan 2021 - Jul 2026), including each day's prior close so
overnight gaps are measurable.

Four components in `src/indicators/williams.py`, each measured **standalone on
train** before anything is combined:

| Component | Entry |
|-----------|-------|
| `gsv` | Buy stop at `open + GSV_buy x multiplier`, sell stop at `open - GSV_sell x multiplier`. `GSV_buy` averages `open - low` over the last N up-closing days; `GSV_sell` averages `high - open` over the last N down-closing days. |
| `oops` | Open below yesterday's low, buy stop at yesterday's low (mirrored for shorts). |
| `smash` | Yesterday closed below the prior N-day low, buy stop at yesterday's high (mirrored). |
| `tdom` | Mean open-to-close return by trading-day-of-month index, **fitted on train only** and then frozen. Used as a bias/filter, never as a system. |

Exits are fixed points plus a day-count exit: `stop_points`, `target_points`
(fixed at 2x the stop, declared not tuned), `max_hold_days` counting the entry
day as day 1, and an optional trailing stop on daily closes.

`src/backtester/daily_engine.py` is a separate engine because the 5-minute
engine enters at a bar's close and cannot represent a resting stop order. It
models, in this order for every day: an **overnight gap through the stop**
(stops are day orders, so the fill is the next session's open, not the stop),
the intraday path (1-minute session paths locate the entry and exit minute, so
stop-versus-target ordering is resolved rather than assumed; ties inside one
minute go to the stop), the stop-order entry, the day-count exit, and the
optional trailing exit.

### Anti-overfitting

- **Split** - straight chronological, with **5 trading days discarded between
  segments** to kill autocorrelation leakage: train 50% / embargo / validation
  20% / embargo / OOS ~30%. All five segments' date ranges and day counts are
  printed. This is deliberately different from the interleaved-block split used
  for the 5-minute study.
- **Grid** - hard-capped at 40 combinations
  (`src/analysis/daily_selection.py` raises above that). The grid is
  `gsv_lookback x gsv_multiplier x stop_points x max_hold_days` = 36. Scored on
  train, top 3 to validation, one locked, OOS touched once.
- **Component selection** - a component is a candidate only if it is profitable
  standalone on train; combinations are then compared on train and the winner
  is reported with its reason.
- **Look-ahead** - `tests/test_lookahead.py` asserts that GSV averages, all
  three trigger sets, the TDOM table and the strategy's signals at bar *i* are
  unchanged when every bar after *i* is deleted or replaced with noise.
- **COT** - `src/data/cot.py` fetches the CFTC legacy Commitments of Traders
  history and builds Williams' COT Index (commercial net position min-max
  normalized over 156 weeks, mapped to daily bars with the Tuesday-to-Friday
  publication lag). It is reported as a diagnostic only; if the fetch fails the
  pipeline says plainly that COT is absent.

## The Baseline Strategy

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

### Interleaved-block split (5-minute study only)

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
config/default.yaml              All parameters, grids and prop firm rules
src/
  strategies/
    williams_strategy.py         PRIMARY: daily Williams components
    simple_strategy.py           5-min baseline (level + delta)
    order_flow_strategy.py       baseline
    volume_profile_strategy.py   baseline
    base.py
  backtester/
    daily_engine.py              Daily stop-entry engine with gap fills
    embargo_split.py             Chronological split with embargo gaps
    engine.py                    5-min bar-by-bar simulation
    block_split.py               Interleaved-block split and per-block runs
    costs.py                     Volatility-scaled slippage + commission
  analysis/
    daily_selection.py           Capped grid: train -> validation -> lock
    parameter_selection.py       Same, for the 5-min study
    prop_firm.py                 FN 50K sizing, MC, arithmetic ceiling
    walk_forward.py              Rolling walk-forward windows
    metrics.py, monte_carlo.py
  data/
    daily_bars.py                1-min -> daily RTH bars, cached
    cot.py                       CFTC Commitments of Traders + COT Index
    fetcher.py, preprocessor.py, trade_aggregator.py, importer.py
  indicators/                    williams.py, order_flow.py, volume_profile.py
  reports/generator.py           Plotly HTML reports
  williams_pipeline.py           Williams study orchestrator
  main.py                        Runs both studies
tests/                           Unit, look-ahead and split regression tests
```

## Known Caveats

These are printed by the pipeline as well:

- **Overnight gaps dominate the daily system.** The median overnight gap on NQ
  is ~69 points, larger than any stop the grid can choose. Gap fills on OOS
  averaged ~98 points worse than the intended stop. This is the reason a daily
  system and a $1,000 daily loss limit are close to incompatible.
- **COT is absent from the traded system.** It is fetched and measured but did
  not improve train P&L, so it was not adopted. Williams treated commercial
  positioning as his primary bias, so only his timing tools are validated here.
- **Overnight holds in the 5-minute study.** Bars outside 09:30-16:00 ET are
  filtered out before that strategy runs, so a position held past the close
  never has its stop tested against the overnight tape. The pipeline reports how
  many trades and how much P&L this affects.
- **Month confounding.** See the interleaved-block split section above.
- **The contract cap dominates the return question.** Because the maximum size
  does not grow with equity, returns accumulate linearly, not exponentially.

## License

MIT
