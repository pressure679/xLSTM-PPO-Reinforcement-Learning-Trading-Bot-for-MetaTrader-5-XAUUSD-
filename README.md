# MT5 XAUUSD LSTM PPO Trading Bot

A MetaTrader 5 reinforcement learning trading bot for XAUUSD (Gold).

The bot combines an LSTM neural network for sequence learning with Proximal Policy Optimization (PPO) to learn trading decisions directly from historical market data.

Unlike traditional bots that rely on fixed rules, the PPO agent learns when to Buy, Sell or Hold from thousands of market examples.

The program should be in a folder or the desktop where the 5m csv file with OHLC data from https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024 is, in training it will then make a LSTM-PPO-saves folder where the training is saved (used for making decisions, also in testing).

---

## Features

### Reinforcement Learning

- LSTM policy network
- PPO training
- Continuous online training
- Live inference on MT5
- Automatic checkpoint saving/loading

### Technical Indicators

- EMA 7
- EMA 21
- EMA Difference (Momentum)
- ADX
- +DI
- -DI
- Stochastic
- VWAP
- VWAP Bands
- VWAP Slope
- Volume Moving Average

### Smart Money Concepts

- Bullish Order Blocks
- Bearish Order Blocks
- Bullish Fair Value Gaps
- Bearish Fair Value Gaps
- Bullish Rejection Blocks
- Bearish Rejection Blocks
- Equal Highs
- Equal Lows
- Market Breaks
- Indecision Candles

### PPO State Features

Current state contains:

- OHLC
- EMA trend
- Momentum
- ADX trend strength
- DI Direction
- Stochastic
- VWAP
- VWAP Bands
- VWAP Position
- VWAP Slope
- Volume MA
- Order Blocks
- Fair Value Gaps
- Rejection Blocks
- Equal Highs/Lows
- Buy Score
- Sell Score

---

## Current Strategy

Current reward structure:

- Take Profit: 20 pips
- Stop Loss: 40 pips
- Risk/Reward: 1 : 0.5

The current focus is high-probability momentum trades rather than large swing trades.

---

## Training

The PPO agent trains continuously over historical MT5 data.

Example metrics during training:

- Win rate: 70–85%
- Profit Factor: 1.5–3+
- Weekly performance: typically 10–30R during training (varies by market conditions)

These figures are training statistics only and are not guarantees of future performance.

---

## Requirements

- Python 3.11+
- MetaTrader 5
- MetaTrader5
- pandas
- numpy
- torch

Install dependencies:

```bash
pip install MetaTrader5 pandas numpy torch
```

---

## Running

Train:

```bash
python mt5-xau-lstm-ppo-bot.py --train
```

Live trading:

```bash
python mt5-xau-lstm-ppo-bot.py --test
```

Train and trade simultaneously:

```bash
python mt5-xau-lstm-ppo-bot.py --train --test
```

---

To train download a m5 xauusd csv file form kaggle or dukascopy and place the folder in "download" relative to the directory the mt5-xau-lstm-ppo-bot.py is in.

## Project Goals

Current work focuses on:

- Improving PPO policy learning
- Better feature engineering
- Dynamic trade management
- VWAP and volume analysis
- Smart Money Concept detection
- MFE/MAE prediction research
- Higher timeframe context

---

## Disclaimer

This project is for educational and research purposes only.

Trading leveraged products involves substantial risk. Always test thoroughly on historical data and demo accounts before risking real capital.
