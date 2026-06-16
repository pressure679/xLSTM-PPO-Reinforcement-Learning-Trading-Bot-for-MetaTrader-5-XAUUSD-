# XAUUSD LSTM + PPO Trading Bot

## Overview

This project is an automated trading system for XAUUSD (Gold) that combines a Long Short-Term Memory (LSTM) neural network with a Proximal Policy Optimization (PPO) reinforcement learning agent.

The system is trained on historical 15-minute XAUUSD OHLC data and learns to make trading decisions based on market structure, technical indicators, and recent price action.

The bot can be trained on historical data and deployed for live trading through MetaTrader 5 using the Python MetaTrader5 library.

---

## Features

* LSTM-based market state representation
* PPO reinforcement learning agent
* Automated trade execution through MetaTrader 5
* Dynamic risk management
* Partial take-profit system
* Training performance reporting
* Walk-forward evaluation support

---

## Data

Training data consists of:

* XAUUSD (Gold)
* 5-minute timeframe
* OHLC candles
* Historical dataset obtained from Kaggle - https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024

The dataset is used to train the LSTM and PPO models on 5-1 year worth of data to identify profitable trading opportunities.

---

## Technical Indicators

The model uses the following indicators as features:

### EMA 7

Fast Exponential Moving Average used to detect short-term trend direction.

### EMA 21

Slower Exponential Moving Average used to identify broader trend structure.

### ADX (Average Directional Index)

Measures trend strength and helps distinguish trending markets from ranging markets.

### Stochastic Oscillator

Used to identify momentum shifts and overbought/oversold conditions.

---

## Model Architecture

### LSTM Network

The LSTM processes recent market data sequences and generates feature representations for the PPO agent.

Input features include:

* OHLC data
* EMA 7
* EMA 21
* ADX
* Stochastic Oscillator

The LSTM learns temporal relationships in price movement and indicator behavior.

### PPO Agent

The PPO agent receives observations from the environment and decides:

* Buy
* Sell
* Hold

The agent learns through reward optimization based on trade outcomes.

---

## Risk Management

The bot uses:

* Percentage-based stop losses derived from current market price
* Dynamic position sizing
* Four partial take-profit targets
* Maximum lot size limits
* Minimum lot size enforcement

Position sizing example:

```python
risk_per_position = min(
    max(round(balance * RISK / 500, 2), 0.01),
    100.0
)
```

---

## Live Trading

Live trading is performed using the MetaTrader5 Python package.

The bot:

1. Connects to MetaTrader 5
2. Retrieves live market prices
3. Generates predictions
4. Places orders automatically
5. Manages open positions

Supported broker symbols may vary depending on the broker configuration.

---

## Training Statistics

During training, the bot reports detailed performance metrics:

### Trades

Total number of completed trades.

### Weekly PnL

Profit and loss measured in pips.

### Win Rate

Percentage of winning trades.

### Mean Win

Average profit per winning trade.

### Mean Loss

Average loss per losing trade.

### Profit Factor (PF)

Calculated as:

PF = Gross Profit / Gross Loss

Measures overall profitability.

### Maximum Drawdown (Max DD)

Largest equity decline during the evaluation period.

### R Profit

Risk-adjusted profit measured in units of R.

R Profit normalizes performance relative to stop-loss risk and position scaling.

### Sharpe Ratio

Measures risk-adjusted return using standard deviation.

### Sortino Ratio

Measures risk-adjusted return while only penalizing downside volatility.

---

## Example Training Output

```text
================================================
[XAUUSD] WEEKLY PPO TRAINING
================================================
Trades:          172
Weekly PnL:      880 pips
Winrate:         91.86%
Mean Win:        10 pips
Mean Loss:       -50 pips
Max DD:          2.00R
PF:              2.26
Weekly R PnL:    17.60R
Sharpe:          0.31
Sortino:         0.10
================================================
```

---

## Requirements

Python packages used include:

* pandas
* numpy
* MetaTrader5

Additional packages may be required depending on the training configuration.

---

## Disclaimer

This software is provided for research and educational purposes.

Trading financial markets involves substantial risk and may result in losses. Past performance does not guarantee future results.

Always test thoroughly on historical and demo accounts before using real capital.
