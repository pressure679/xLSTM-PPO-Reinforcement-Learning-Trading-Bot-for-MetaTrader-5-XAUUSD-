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

Computes ICT/SMC-style features per 1-minute candle: EMAs (7/21/50/200) and their
slopes, ADX/DI, stochastic, VWAP (+bands), order blocks (+mitigation), fair value gaps
(+retracement, +inverse FVG), market structure shifts (+retest, +0.786 fib
retracement), break of structure / trend / change of character, breaker blocks,
rejection blocks (+mitigation), equal highs/lows (+retest), support/resistance
trendlines, Fibonacci retracement zones, internal/external liquidity ranges, killzone
(Asia/London/NY), Asia-session/PDH/PDL/POC distances, and two heuristic confluence
scores (`buy_score` / `sell_score`).

It then resamples the same 1-minute candles into 15-minute candles — right-labeled so
a bar is only "visible" once it has actually closed, avoiding lookahead into a
still-forming candle — reruns the *same* indicator pipeline on them, and forward-fills
every resulting column back onto each 1-minute row with a `15m_` prefix (e.g.
`15m_EMA7`, `15m_buy_score`). The model sees both timeframes at once from a single
feature vector per 1-minute bar.

### Agent — `LSTMPPOAgent` / `PPOLSTMNetwork`

An LSTM encoder over a `SEQ_LEN`-bar window of features feeds a PPO actor-critic head
with 3 actions (`hold`, `long`, `short`). The reward for a closed trade is its raw
pnl (in pips).

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

## Notes

- `state_size` (i.e. `len(FEATURES)`) must match whatever a saved checkpoint was
  trained with — after any change to the feature set, retrain from scratch before
  `test_bot` can load the new checkpoint.
- `PDPOCDistance()` expects a `Volume` column; MT5's live feed only has `tick_volume`,
  which `test_bot` renames to `Volume` before calling `add_indicators`.
