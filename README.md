# bot.py — LSTM-PPO XAUUSD Trading Bot

An ICT/SMC-style feature-engineering pipeline feeding an LSTM-PPO reinforcement-learning
agent that trades XAUUSD (gold) via MetaTrader5. One file, two modes: train against
historical 1-minute data, or trade live.

## Requirements

- Windows (the `MetaTrader5` Python package wraps the MT5 terminal API and only works there)
- Python 3.10+
- A running MetaTrader5 terminal, logged into a broker account, with algotrading enabled
- `pip install -r requirements.txt`

## Data

- `download/XAU_1m_data.csv` — historical 1-minute XAUUSD OHLCV, semicolon-delimited
  (`Date;Open;High;Low;Close;Volume`). Required for `--train`.
- `download/dxy_history.csv`, `download/real_yield_history.csv`,
  `download/gld_holdings_history.csv` — cached daily macro series (DXY, real yields, GLD
  ETF holdings). Auto-fetched and cached by `add_macro_features()` on first use (needs
  network access), refreshed after ~20h.
- `LSTM-PPO-saves/` — model checkpoints, named `{date}-{symbol}.checkpoint.pt`.
- `LSTM-PPO-saves/{symbol}.winrate.pkl` — accumulated (feature vector, win/loss)
  samples backing the `WinRateModel` win-rate filter, capped at ~20MB (see
  "Win-rate filter" below).

## Usage

```bash
python bot.py --train                              # train against historical data
python bot.py --test --symbol XAUUSD-STDc           # trade live via MT5
python bot.py --train --test --symbol XAUUSD-STDc   # both at once
```

Both flags spawn daemon threads (`train_bot` / `test_bot`); the process blocks on
`thread.join()` until interrupted (Ctrl+C).

## How it works

### Feature engineering — `add_indicators(df)`

Every 1-minute candle gets the full indicator stack below, then the same stack is
recomputed on 15-minute candles aggregated from that same data and merged back in
(see "Multi-timeframe merge"), so the final feature vector for one 1-minute bar
actually contains both timeframes' worth of state.

**Trend / momentum**
- EMA 7 / 21 / 50 / 200, each with its slope, plus the 7-21 and 50-200 diffs (`EMA`)
- ADX, +DI, -DI (`ADX`)
- Stochastic %K / smoothed %K (`STOCH`)

**Price location / volatility**
- Session VWAP with ±1 stddev bands, above/below flags, and slope (`VWAP`)
- Candle range and its rolling average (`GetRange`, `RangeMA`)
- Indecision (small-body doji-like) candles (`Indecision`)
- Bullish/bearish candle direction (`BullishBearish`)
- Exhaustion candles — body much smaller than the previous one (`ExhaustionCandle`)

**Smart Money Concepts (SMC/ICT) structure**
- Order blocks, bullish & bearish, plus mitigation (`BullishOB`, `BearishOB`, `OBMitigation`)
- Fair value gaps, bullish & bearish, plus retracement into the gap and inverse FVGs
  (`BullishFVG`, `BearishFVG`, `FVGRetracement`, `BullishIFVG`, `BearishIFVG`)
- Market structure shift (MSS) — a close breaking the last swing high/low — plus a
  retest of the broken level and a 0.786 Fibonacci retracement measured from the MSS
  breakout candle back to the swing that anchored it (`MSS`, `MSS_Retest`,
  `MSSFibRetracement` — the retracement variant is specific to this bot, not a
  standard ICT indicator)
- Break of structure, the resulting trend state, and change of character
  (`BoS`, `Trend`, `CHoCH`)
- Breaker blocks — an order block that's been fully mitigated and flipped
  (`BullishBB`, `BearishBB`)
- Rejection blocks (long-wick candles) plus mitigation of the swing they rejected
  (`RejectionBlocks`, `ReversalBlockMitigation`)
- Mitigation blocks — used in the FVG+MB/RB confluence term of the score
  (`BullishMB`, `BearishMB`)
- Equal highs / equal lows (liquidity pools) plus a retest of that level
  (`EQH_EQL`, `EQHEQLRetest`)
- Internal vs. external range liquidity — whether price is inside or has swept
  beyond the current dealing range (`add_irl_erl`)

**Fibonacci**
- Retracement zones (23.6% / 38.2% / 50% / 61.8% / 78.6%) of the last swing-low →
  swing-high (bullish) or swing-high → swing-low (bearish) impulse (`Fibonacci`)

**Trendlines**
- Dynamic ascending support and descending resistance lines drawn through the two
  most recent conforming swing lows / swing highs (searching back past any
  non-conforming swing in between), with "is price above support / below
  resistance" flags (`TrendLines`)

**Session & liquidity distances**
- Killzone: Asia / London / New York session flag (`GetKillzone`)
- Distance from the current Asia session high/low (`AsiaHighDistance`, `AsiaLowDistance`)
- Distance from the previous day's high/low and volume point of control
  (`PDHDistance`, `PDLDistance`, `PDPOCDistance`)

**Composite scores**
- `buy_score` / `sell_score` — a weighted sum of the confluences above (trend
  alignment, OB mitigation, breaker blocks, MSS/CHoCH, FVG+MB/RB confluence,
  liquidity distances, equal-high/low retests, Fibonacci-in-trend, trendline
  position), gated to only fire inside a killzone and above `adx=20` (`BuyScore`,
  `SellScore`)

**Multi-timeframe merge**
- The raw 1-minute OHLCV is resampled into 15-minute candles — right-labeled so a bin
  covering `[10:00, 10:15)` is only "visible" starting at 10:15, i.e. once it has
  actually closed — and the entire stack above is rerun on those 15-minute candles.
  Every resulting column is forward-filled back onto each 1-minute row with a `15m_`
  prefix (e.g. `15m_EMA7`, `15m_bullish_mss`, `15m_buy_score`), so no 1-minute bar
  ever sees a still-forming 15-minute candle's data.

**Macro (daily, added separately by `add_macro_features`, not part of `add_indicators`)**
- DXY, real yield, and GLD-holdings-flow z-scores (`dxy_zscore`, `real_yield_zscore`,
  `gld_flow_zscore`)

### Agent — `LSTMPPOAgent` / `PPOLSTMNetwork`

An LSTM encoder over a `SEQ_LEN`-bar window of features feeds a PPO actor-critic head
with 3 actions (`hold`, `long`, `short`). The reward for a closed trade is its raw
pnl (in pips).

### Win-rate filter — `WinRateModel`

A second, independent gate sits between the PPO agent's action and the market. Before
a `long`/`short` action is allowed to actually open a position, an `XGBClassifier`
predicts the probability that a setup like this one — the same single-bar feature
vector the agent just saw — will win, based on every closed trade's entry-time
features and outcome seen so far. If the predicted win rate doesn't clear
`MIN_WINRATE`, the action is silently downgraded to `hold` instead.

- **Training (`train_bot`)**: every closed trade's entry-time state and win/loss
  outcome is recorded (`add_sample`). At the end of each simulated week, the model is
  refit on every sample accumulated so far and persisted (`fit`, `save`) alongside
  that week's checkpoint.
- **Live trading (`test_bot`)**: loads the persisted model at startup and applies it
  read-only — it never fits or saves during live trading, only `train_bot` does.
- **Bootstrapping**: before a saved file exists, or before `min_samples` trades have
  accumulated, the filter is a no-op — every action passes through unfiltered — so
  early training and first live runs aren't blocked waiting on data that doesn't
  exist yet.
- **Persistence**: raw `(X, y)` samples, not the fitted model itself, are pickled to
  `LSTM-PPO-saves/{symbol}.winrate.pkl`, so the sample set survives an algorithm
  change and keeps accumulating indefinitely — capped at `max_mb` (default 20MB) by
  trimming to the most recently seen samples once the pickled size exceeds it.

### Training — `train_bot()`

- Loads the CSV, adds indicators, adds macro z-score features (`dxy_zscore`,
  `real_yield_zscore`, `gld_flow_zscore`).
- Simulates trading bar-by-bar with a fixed take-profit (`TP_PIPS = 20`) and a
  recovery-factor-throttled stop-loss (`BASE_SL_PIPS = 40`, scaled down to as little as
  `MIN_SL_SCALE = 1/3` of that when the trailing 50-trade recovery factor drops below
  `TARGET_RF = 1.0`).
- Every simulated week (`save_count = 1440 * 5` one-minute bars ≈ 5 trading days):
  preloads that week's feature rows as one vectorized slice, prints a weekly report
  (trade count, PnL, winrate, mean win/loss, max drawdown, profit factor, recovery
  factor, Sharpe, Sortino, Z-score, average win/loss streak), trains the PPO agent,
  unconditionally clears its trajectory buffer, and saves a checkpoint.
- Loops indefinitely, re-running the full dataset each "pass," until interrupted.

### Live trading — `test_bot(symbol)`

Connects to MetaTrader5, polls for new 1-minute candles, recomputes indicators on a
rolling ~3000-bar window each time a candle closes, and feeds the trained agent's
action into `open_long()` / `open_short()` / `close_trades()` / `modify_sl()`.

## Configuration

Key constants (edit in-file — there's no config file):

| Constant | Where | Purpose |
|---|---|---|
| `SEQ_LEN` | `train_bot`, `test_bot` | LSTM lookback window, in bars |
| `TP_PIPS` / `BASE_SL_PIPS` | `train_bot`, `test_bot` | Fixed take-profit / base stop-loss |
| `TARGET_RF` / `MIN_SL_SCALE` | `train_bot`, `test_bot` | Recovery-factor stop-loss throttle |
| `save_count` | `train_bot` | Bars per simulated "week" (report/train/checkpoint cadence) |
| `magic=123456` | throughout | MT5 order/position tag identifying this EA's trades |
| `MIN_WINRATE` | `train_bot`, `test_bot` | Win-rate filter threshold a setup's predicted win probability must clear to open a trade (currently hardcoded to `0.9`; the commented-out alternative is the RR-implied breakeven, `1 / (1 + RR_RATIO)`) |
| `min_samples` / `max_mb` | `WinRateModel.__init__` | Minimum accumulated trades before the win-rate filter starts predicting / max `.winrate.pkl` size before oldest samples are trimmed |

## Notes

- `state_size` (i.e. `len(FEATURES)`) must match whatever a saved checkpoint was
  trained with — after any change to the feature set, retrain from scratch before
  `test_bot` can load the new checkpoint. The same applies to `{symbol}.winrate.pkl`:
  its stored feature vectors are shaped by whatever `FEATURES` was at the time each
  sample was recorded, so a feature-set change also makes the accumulated win-rate
  samples stale and it should be rebuilt from a fresh `train_bot` run.
- `PDPOCDistance()` expects a `Volume` column; MT5's live feed only has `tick_volume`,
  which `test_bot` renames to `Volume` before calling `add_indicators`.
