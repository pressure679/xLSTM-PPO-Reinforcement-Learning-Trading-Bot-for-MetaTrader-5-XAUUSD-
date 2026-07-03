import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO
import random
from collections import deque
from datetime import datetime, timedelta
import subprocess
import time
import argparse
import threading
import math
import glob
import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
"""
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    FeedForwardConfig,
)
"""
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)

ACTIONS = ['hold', 'long', 'short']

def find_latest_dukascopy_csv():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download")

    pattern = os.path.join(folder, "xauusd-m5-bid-*.csv")

    # print(pattern)
    # print(glob.glob(pattern))
    # print(os.listdir(folder))

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No CSV found matching {pattern}")

    return max(files, key=os.path.getmtime)

def load_last_mb_xauusd(file_path=None, mb=8, delimiter=',', col_names=None):
    if file_path is None:
        file_path = find_latest_dukascopy_csv()

    print(f"Loading: {file_path}")

    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)  # start position
    
    with open(file_path, 'rb') as f:
        # Seek to approximately 20 MB before EOF
        f.seek(offset)
        
        # Read to the end of file from that offset
        data = f.read().decode(errors='ignore')
        
        # If not at start of file, discard partial first line (incomplete)
        if offset > 0:
            data = data.split('\n', 1)[-1]
    
    """
    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=None, engine='python')
    
    #if col_names:
    print(df.head())
    print(df.columns)
    print(df.shape)

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    """

    df = pd.read_csv(
        StringIO(data),
        delimiter=delimiter,
        header=0,
        engine="python"
    )

    """
    df.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)
    """

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Convert columns if needed, e.g.:
    # df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors='coerce')
    df["Date"] = pd.to_datetime(
        df["Date"],
        unit="ms",
        utc=True
    )
    # print(df.columns.tolist())
    # print(df.head())
    # for col in ["Open", "High", "Low", "Close", "Volume"]:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    # df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()

    print(f"Loaded: {file_path}")
    
    df = df.dropna()
    
    return df

def ADX(df, period=14):
    """
    Returns +DI, -DI and ADX using Wilder's smoothing.
    Columns required: High, Low, Close
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- directional movement -----------------------------------------
    # plus_dm  = (high.diff()  > low.diff())  * (high.diff()).clip(lower=0)
    # minus_dm = (low.diff()   > high.diff()) * (low.diff().abs()).clip(lower=0)̈́
    up  =  high.diff()
    dn  = -low.diff()

    plus_dm_array  = np.where((up  >  dn) & (up  > 0),  up,  0.0)
    minus_dm_array = np.where((dn  >  up) & (dn  > 0),  dn,  0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index) # ← wrap
    minus_dm = pd.Series(minus_dm_array, index=df.index) # ← wrap

    # --- true range ----------------------------------------------------
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # --- Wilder smoothing ---------------------------------------------
    atr       = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di  = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx = round(adx , 2)
    plus_di = round(plus_di, 2)
    minus_di = round(minus_di, 2)

    return adx, plus_di, minus_di

def STOCH(df, period=14, smooth_d=3):
    """
    Returns %K and %D stochastic oscillator.

    Columns required:
    High, Low, Close
    """

    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- highest high / lowest low ------------------------------------
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    # --- %K ------------------------------------------------------------
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

    # --- %D (smoothed %K) ---------------------------------------------
    d = k.rolling(window=smooth_d).mean()

    k = round(k, 2)
    d = round(d, 2)

    return k, d

def EMA(df, period):
    return df['Close'].ewm(span=period, adjust=False).mean().round(2)

def EQH_EQL(df, threshold=10, lookback=72, min_distance=3):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    eqh = np.zeros(len(df), dtype=np.bool_)
    eql = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - min_distance, start - 1, -1):

            # Equal High
            if (
                abs(high[i] - high[j]) <= threshold or
                abs(high[i] - body_high[j]) <= threshold or
                abs(body_high[i] - high[j]) <= threshold or
                abs(body_high[i] - body_high[j]) <= threshold
            ):
                eqh[i] = True

            # Equal Low
            if (
                abs(low[i] - low[j]) <= threshold or
                abs(low[i] - body_low[j]) <= threshold or
                abs(body_low[i] - low[j]) <= threshold or
                abs(body_low[i] - body_low[j]) <= threshold
            ):
                eql[i] = True

            if eqh[i] and eql[i]:
                break

    return eqh, eql

def EQHEQLRetest(df, threshold=20, lookback=12):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    eqh = df["eqh"].to_numpy()
    eql = df["eql"].to_numpy()

    eqh_retest = np.zeros(len(df), dtype=np.bool_)
    eql_retest = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        # Previous EQH
        for j in range(i - 1, start - 1, -1):

            if not eqh[j]:
                continue

            level = max(high[j], body_high[j])

            if (
                abs(high[i] - level) <= threshold or
                abs(body_high[i] - level) <= threshold
                # and df["bullish"].iloc[i]
            ):
                eqh_retest[i] = True
                break

        # Previous EQL
        for j in range(i - 1, start - 1, -1):

            if not eql[j]:
                continue

            level = min(low[j], body_low[j])

            if (
                (
                    abs(low[i] - level) <= threshold or
                    abs(body_low[i] - level) <= threshold
                )
                and df["bearish"].iloc[i]
            ):
                eql_retest[i] = True
                break

    return eqh_retest, eql_retest
def Indecision(df, threshold=0.2):
    body = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, 1e-9)

    return (body / candle_range < threshold).astype(int)

def RejectionBlocks(df, wick_ratio=2.0):
    body = (df["Close"] - df["Open"]).abs()

    upper = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - df["Low"]

    bullish_rb = (
        (lower > body * wick_ratio) &
        (lower > upper)
    ).astype(int)

    bearish_rb = (
        (upper > body * wick_ratio) &
        (upper > lower)
    ).astype(int)

    return bullish_rb, bearish_rb

def BullishOB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)

    return (
        bearish &
        next_bullish &
        (body > df["range_ma"]) &
        (next_body >= body * multiplier)
    ).astype(int)

def BearishOB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)

    return (
        bullish &
        next_bearish &
        (body > df["range_ma"]) &
        (next_body >= body * multiplier)
    ).astype(int)

def BullishFVG(df):
    return (df["Low"].shift(-1) > df["High"].shift(1)).astype(int)

def BearishFVG(df):
    return (df["High"].shift(-1) < df["Low"].shift(1)).astype(int)

def BullishIFVG(df):
    return (
        (df["bearish_fvg"].shift(1) == 1) &
        (df["Close"] > df["Low"].shift(2))
    ).astype(int)

def BearishIFVG(df):
    return (
        (df["bullish_fvg"].shift(1) == 1) &
        (df["Close"] < df["High"].shift(2))
    ).astype(int)

def BullishMB(df, multiplier=1.0):
    body = (df["Close"] - df["Open"]).abs()

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)
    displacement = body.shift(-1) >= body * multiplier

    ob = bearish & next_bullish & displacement

    ob_high = df["High"].where(ob).ffill()
    ob_low = df["Low"].where(ob).ffill()

    return ((df["Low"] <= ob_high) &
            (df["High"] >= ob_low)).astype(int)

def BearishMB(df, multiplier=1.0):
    body = (df["Close"] - df["Open"]).abs()

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)
    displacement = body.shift(-1) >= body * multiplier

    ob = bullish & next_bearish & displacement

    ob_high = df["High"].where(ob).ffill()
    ob_low = df["Low"].where(ob).ffill()

    return ((df["Low"] <= ob_high) &
            (df["High"] >= ob_low)).astype(int)

def Swings(df, left=2, right=2):

    body_high = np.maximum(df["Open"].to_numpy(), df["Close"].to_numpy())
    body_low = np.minimum(df["Open"].to_numpy(), df["Close"].to_numpy())

    swing_high = [0] * len(df)
    swing_low = [0] * len(df)

    for i in range(left, len(df) - right):

        # Swing High (body only)
        if (
            body_high[i] > body_high[i-1] and
            body_high[i] > body_high[i-2] and
            body_high[i] > body_high[i+1] and
            body_high[i] > body_high[i+2]
        ):
            swing_high[i] = 1

        # Swing Low (body only)
        if (
            body_low[i] < body_low[i-1] and
            body_low[i] < body_low[i-2] and
            body_low[i] < body_low[i+1] and
            body_low[i] < body_low[i+2]
        ):
            swing_low[i] = 1

    return swing_high, swing_low

def MSS(df):

    bullish = pd.Series(0, index=df.index)
    bearish = pd.Series(0, index=df.index)

    last_swing_high = None
    last_swing_low = None

    for i in range(len(df)):

        if df["swing_high"].iloc[i]:
            last_swing_high = df["High"].iloc[i]

        if df["swing_low"].iloc[i]:
            last_swing_low = df["Low"].iloc[i]

        if (
            last_swing_high is not None
            and df["Close"].iloc[i] > last_swing_high
        ):
            bullish.iloc[i] = 1
            last_swing_high = None

        if (
            last_swing_low is not None
            and df["Close"].iloc[i] < last_swing_low
        ):
            bearish.iloc[i] = 1
            last_swing_low = None

    return bullish, bearish

def MSS_Retest(df, lookback=72, threshold=3):

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    bullish_mss = df["bullish_mss"].to_numpy()
    bearish_mss = df["bearish_mss"].to_numpy()

    bullish_retest = np.zeros(len(df), dtype=np.int8)
    bearish_retest = np.zeros(len(df), dtype=np.int8)

    last_bull_level = None
    last_bear_level = None
    last_bull_index = -1
    last_bear_index = -1

    for i in range(len(df)):

        # Remember bullish MSS level
        if bullish_mss[i]:
            last_bull_level = high[i]
            last_bull_index = i

        # Remember bearish MSS level
        if bearish_mss[i]:
            last_bear_level = low[i]
            last_bear_index = i

        # Bullish retest
        if (
            last_bull_level is not None
            and i - last_bull_index <= lookback
            and low[i] <= last_bull_level + threshold
            and high[i] >= last_bull_level - threshold
        ):
            bullish_retest[i] = 1
            last_bull_level = None

        # Bearish retest
        if (
            last_bear_level is not None
            and i - last_bear_index <= lookback
            and high[i] >= last_bear_level - threshold
            and low[i] <= last_bear_level + threshold
        ):
            bearish_retest[i] = 1
            last_bear_level = None

    return bullish_retest, bearish_retest

def BoS(df):

    bullish_bos = np.zeros(len(df), dtype=np.int8)
    bearish_bos = np.zeros(len(df), dtype=np.int8)

    last_high = None
    last_low = None

    trend = None

    for i in range(len(df)):

        # New swing high
        if df["swing_high"].iat[i]:

            high = df["High"].iat[i]

            if last_high is not None:

                if high > last_high:
                    # Higher High
                    if trend == "bull":
                        bullish_bos[i] = 1
                    trend = "bull"

                elif high < last_high:
                    # Lower High
                    trend = "bear"

            last_high = high

        # New swing low
        if df["swing_low"].iat[i]:

            low = df["Low"].iat[i]

            if last_low is not None:

                if low < last_low:
                    # Lower Low
                    if trend == "bear":
                        bearish_bos[i] = 1
                    trend = "bear"

                elif low > last_low:
                    # Higher Low
                    trend = "bull"

            last_low = low

    return bullish_bos, bearish_bos

def Trend(df):

    bullish_trend = np.zeros(len(df), dtype=np.int8)
    bearish_trend = np.zeros(len(df), dtype=np.int8)

    bull_count = 0
    bear_count = 0

    trend = 0
    # 0 = none
    # 1 = bullish
    # -1 = bearish

    for i in range(len(df)):

        if df["bullish_bos"].iat[i]:
            bull_count += 1
            bear_count = 0

            if bull_count >= 2:
                trend = 1

        elif df["bearish_bos"].iat[i]:
            bear_count += 1
            bull_count = 0

            if bear_count >= 2:
                trend = -1

        if trend == 1:
            bullish_trend[i] = 1
        elif trend == -1:
            bearish_trend[i] = 1

    return bullish_trend, bearish_trend

def CHoCH(df, lookback=72):

    bullish_choch = np.zeros(len(df), dtype=np.int8)
    bearish_choch = np.zeros(len(df), dtype=np.int8)

    for i in range(lookback, len(df)):

        start = max(0, i - lookback)

        # ---------- Bullish trend -> look for bearish CHoCH ----------
        if df["bullish_trend"].iat[i]:

            last_hl = None

            for j in range(i - 1, start - 1, -1):
                if df["swing_low"].iat[j]:
                    last_hl = df["Low"].iat[j]
                    break

            if last_hl is not None and df["Close"].iat[i] < last_hl:
                bearish_choch[i] = 1

        # ---------- Bearish trend -> look for bullish CHoCH ----------
        elif df["bearish_trend"].iat[i]:

            last_lh = None

            for j in range(i - 1, start - 1, -1):
                if df["swing_high"].iat[j]:
                    last_lh = df["High"].iat[j]
                    break

            if last_lh is not None and df["Close"].iat[i] > last_lh:
                bullish_choch[i] = 1

    return bullish_choch, bearish_choch

def BullishBB(df, lookback=72):

    bb = np.zeros(len(df), dtype=np.int8)

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    bearish_ob = df["bearish_ob"].to_numpy()

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):

            if bearish_ob[j] != 1:
                continue

            ob_high = high[j]
            ob_low = low[j]

            # Highest high and lowest low since the OB formed
            highest = np.max(high[j+1:i+1])
            lowest = np.min(low[j+1:i+1])

            if highest >= ob_high and lowest <= ob_low:
                bb[i] = 1
                break

    return bb

def BearishBB(df, lookback=72):

    bb = np.zeros(len(df), dtype=np.int8)

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    bullish_ob = df["bullish_ob"].to_numpy()

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):

            if bullish_ob[j] != 1:
                continue

            ob_high = high[j]
            ob_low = low[j]

            # Highest high and lowest low since the OB formed
            highest = np.max(high[j + 1:i + 1])
            lowest = np.min(low[j + 1:i + 1])

            # Order block has been fully mitigated
            if highest >= ob_high and lowest <= ob_low:
                bb[i] = 1
                break

    return bb

def ExhaustionCandle(df):
    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body = np.abs(close - open_)

    exhaustion = np.zeros(len(df), dtype=np.bool_)

    exhaustion[1:] = body[1:] <= body[:-1] * 0.10

    return exhaustion

def OBMitigation(df, threshold=30, lookback=72):

    low = df["Low"].to_numpy()
    high = df["High"].to_numpy()

    bullish_ob = df["bullish_ob"].to_numpy()
    bearish_ob = df["bearish_ob"].to_numpy()

    bull = np.zeros(len(df), dtype=np.bool_)
    bear = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):

            if not np.isnan(bullish_ob[j]):

                if abs(low[i] - bullish_ob[j]) <= threshold:
                    bull[i] = True
                    break

        for j in range(i - 1, start - 1, -1):

            if not np.isnan(bearish_ob[j]):

                if abs(high[i] - bearish_ob[j]) <= threshold:
                    bear[i] = True
                    break

    return bull, bear

def TrendLines(df):

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    above_support = np.zeros(len(df), dtype=np.bool_)
    below_resistance = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    # =====================================================
    # Support
    # =====================================================

    lows = np.where(swing_low)[0]

    for k in range(1, len(lows)):

        i1 = lows[k - 1]
        i2 = lows[k]

        # Only higher lows
        if low[i2] <= low[i1]:
            continue

        slope = (low[i2] - low[i1]) / (i2 - i1)

        end = lows[k + 1] if k + 1 < len(lows) else n

        for j in range(i2 + 1, end):

            trend_price = low[i2] + slope * (j - i2)

            if close[j] >= trend_price:
                above_support[j] = True

    # =====================================================
    # Resistance
    # =====================================================

    highs = np.where(swing_high)[0]

    for k in range(1, len(highs)):

        i1 = highs[k - 1]
        i2 = highs[k]

        # Only lower highs
        if high[i2] >= high[i1]:
            continue

        slope = (high[i2] - high[i1]) / (i2 - i1)

        end = highs[k + 1] if k + 1 < len(highs) else n

        for j in range(i2 + 1, end):

            trend_price = high[i2] + slope * (j - i2)

            if close[j] <= trend_price:
                below_resistance[j] = True

    return above_support, below_resistance

def Fibonacci(df, min_swing=7.5, tolerance=0.05):

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    bullish_fib382 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib500 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib618 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib786 = np.zeros(len(df), dtype=np.bool_)

    bearish_fib382 = np.zeros(len(df), dtype=np.bool_)
    bearish_fib500 = np.zeros(len(df), dtype=np.bool_)
    bearish_fib618 = np.zeros(len(df), dtype=np.bool_)
    bearish_fib786 = np.zeros(len(df), dtype=np.bool_)

    last_type = None
    last_index = None

    n = len(df)

    for i in range(n):

        # ==========================================================
        # Bullish impulse (Swing Low -> Swing High)
        # ==========================================================

        if swing_high[i] and last_type == "low":

            swing = high[i] - low[last_index]

            if swing >= min_swing:

                fib382 = high[i] - swing * 0.382
                fib500 = high[i] - swing * 0.500
                fib618 = high[i] - swing * 0.618
                fib786 = high[i] - swing * 0.786
                threshold = swing * tolerance

                j = i

                while (
                    j < n and
                    not swing_high[j] and
                    not swing_low[j]
                ):

                    bullish_fib382[j] = (
                        body_low[j] <= fib382 + threshold and
                        body_high[j] >= fib382 - threshold
                    )

                    bullish_fib500[j] = (
                        body_low[j] <= fib500 + threshold and
                        body_high[j] >= fib500 - threshold
                    )

                    bullish_fib618[j] = (
                        body_low[j] <= fib618 + threshold and
                        body_high[j] >= fib618 - threshold
                    )

                    bullish_fib786[j] = (
                        body_low[j] <= fib786 + threshold and
                        body_high[j] >= fib786 - threshold
                    )

                    j += 1

            last_type = "high"
            last_index = i

        # ==========================================================
        # Bearish impulse (Swing High -> Swing Low)
        # ==========================================================

        elif swing_low[i] and last_type == "high":

            swing = high[last_index] - low[i]

            if swing >= min_swing:

                fib382 = low[i] + swing * 0.382
                fib500 = low[i] + swing * 0.500
                fib618 = low[i] + swing * 0.618
                fib786 = low[i] + swing * 0.786

                j = i

                while (
                    j < n and
                    not swing_high[j] and
                    not swing_low[j]
                ):

                    bearish_fib382[j] = (
                        body_low[j] <= fib382 + threshold and
                        body_high[j] >= fib382 - threshold
                    )

                    bearish_fib500[j] = (
                        body_low[j] <= fib500 + threshold and
                        body_high[j] >= fib500 - threshold
                    )

                    bearish_fib618[j] = (
                        body_low[j] <= fib618 + threshold and
                        body_high[j] >= fib618 - threshold
                    )

                    bearish_fib786[j] = (
                        body_low[j] <= fib786 + threshold and
                        body_high[j] >= fib786 - threshold
                    )

                    j += 1

            last_type = "low"
            last_index = i

        elif swing_low[i]:
            last_type = "low"
            last_index = i

        elif swing_high[i]:
            last_type = "high"
            last_index = i

    return (
        bullish_fib382,
        bullish_fib500,
        bullish_fib618,
        bullish_fib786,
        bearish_fib382,
        bearish_fib500,
        bearish_fib618,
        bearish_fib786,
    )

def FVGRetracement(df, tolerance=0.05):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    body_mid = (body_high + body_low) / 2

    bull_fvg = df["bullish_fvg"].to_numpy()
    bear_fvg = df["bearish_fvg"].to_numpy()

    bull_ret = np.zeros(len(df), dtype=np.bool_)
    bear_ret = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    # -------------------------
    # Bullish FVG
    # -------------------------
    for i in range(1, n - 1):

        if not bull_fvg[i]:
            continue

        # Fib: 0 = first candle low, 1 = third candle high
        swing_low = df["Low"].iat[i - 1]
        swing_high = df["High"].iat[i + 1]

        height = swing_high - swing_low
        if height <= 0:
            continue

        fib382 = swing_high - 0.382 * height
        fib500 = swing_high - 0.500 * height

        tol = height * tolerance

        for j in range(i + 2, n):

            # Stop when bullish continuation resumes
            if close[j] >= open_[j]:
                break

            if (
                abs(body_mid[j] - fib382) <= tol or
                abs(body_mid[j] - fib500) <= tol
            ):
                bull_ret[j] = True
                break

    # -------------------------
    # Bearish FVG
    # -------------------------
    for i in range(1, n - 1):

        if not bear_fvg[i]:
            continue

        # Fib: 0 = first candle high, 1 = third candle low
        swing_high = df["High"].iat[i - 1]
        swing_low = df["Low"].iat[i + 1]

        height = swing_high - swing_low
        if height <= 0:
            continue

        fib382 = swing_low + 0.382 * height
        fib500 = swing_low + 0.500 * height

        tol = height * tolerance

        for j in range(i + 2, n):

            # Stop when bearish continuation resumes
            if close[j] <= open_[j]:
                break

            if (
                abs(body_mid[j] - fib382) <= tol or
                abs(body_mid[j] - fib500) <= tol
            ):
                bear_ret[j] = True
                break

    return bull_ret, bear_ret

def AsiaHighDistance(df):
    # Asia session: 01:00-08:59 (your timezone)
    asia = (df.index.hour >= 1) | (df.index.hour <= 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_high = (
        df["High"]
        .where(asia)
        .groupby(trade_day)
        .transform("max")
        .ffill()
    )

    return asia_high - df["Close"]

def AsiaLowDistance(df):
    asia = (df.index.hour >= 1) | (df.index.hour <= 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_low = (
        df["Low"]
        .where(asia)
        .groupby(trade_day)
        .transform("min")
        .ffill()
    )

    return df["Close"] - asia_low

def PDHDistance(df):
    day = df.index.date

    daily_high = (
        df["High"]
        .groupby(day)
        .transform("max")
    )

    pdh = (
        daily_high
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return pdh - df["Close"]

def PDLDistance(df):
    day = df.index.date

    daily_low = (
        df["Low"]
        .groupby(day)
        .transform("min")
    )

    pdl = (
        daily_low
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return df["Close"] - pdl

def PDPOCDistance(df, bins=50):
    day = df.index.date
    poc = np.full(len(df), np.nan)

    unique_days = np.unique(day)

    prev_poc = np.nan

    for d in unique_days:
        # Assign previous day's POC to today's candles
        mask = day == d
        poc[mask] = prev_poc

        # Compute today's POC for use tomorrow
        today = df.loc[mask]

        if len(today) > 1:
            prices = ((today["High"] + today["Low"] + today["Close"]) / 3).values
            volumes = today["Volume"].values

            hist, edges = np.histogram(
                prices,
                bins=bins,
                weights=volumes
            )

            idx = np.argmax(hist)
            prev_poc = (edges[idx] + edges[idx + 1]) / 2

    return df["Close"] - poc

def VWAP(df):

    # print(type(df.index))
    # print(df.index.dtype)
    # print(df.index[:5])

    # --------------------------------------------------
    # Select volume column
    # --------------------------------------------------
    if "Volume" in df.columns:
        volume = df["Volume"]
    elif "tick_volume" in df.columns:
        volume = df["tick_volume"]
    elif "real_volume" in df.columns:
        volume = df["real_volume"]
    else:
        raise ValueError("No volume column found.")

    # --------------------------------------------------
    # VWAP
    # --------------------------------------------------
    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    # print(df.columns)
    # df["Date"] = (
    #     pd.to_datetime(df["Date"], unit="ms", utc=True)
    #     .dt.tz_convert("Europe/Denmark")
    #     .dt.tz_localize(None)
    # )
    # index = df.index.tz_convert("Europe/Copenhagen")
    # print(f"vwap time: {df.index[-1]}")

    session = df.index.normalize()
    # session = (df.index - pd.Timedelta(hours=1)).normalize()

    cum_tpv = (typical_price * volume).groupby(session).cumsum()
    cum_volume = volume.groupby(session).cumsum()

    vwap = round(cum_tpv / cum_volume, 2)

    # --------------------------------------------------
    # Session VWAP Standard Deviation
    # --------------------------------------------------

    # Squared distance from VWAP
    sq_diff = ((typical_price - vwap) ** 2) * volume

    # Cumulative weighted variance
    cum_sq_diff = sq_diff.groupby(session).cumsum()

    variance = cum_sq_diff / cum_volume
    stddev = variance.pow(0.5)

    upper = round(vwap + stddev, 2)
    lower = round(vwap - stddev, 2)

    # --------------------------------------------------
    # Derived features
    # --------------------------------------------------
    # dist = df["Close"] - vwap

    above = (df["Close"] > vwap).astype(int)
    below = (df["Close"] < vwap).astype(int)

    above_upper = (df["Close"] > upper).astype(int)
    below_lower = (df["Close"] < lower).astype(int)

    slope = vwap.diff()

    return (
        vwap,
        upper,
        lower,
        # dist,
        above,
        below,
        above_upper,
        below_lower,
        slope
    )

def GetRange(df):
    """
    Returns normalized candle range as a percentage of price.
    """
    return (df["High"] - df["Low"]) * 10

def RangeMA(df, period=14):

    # -----------------------------------------
    # Select volume column
    # -----------------------------------------
    # if "volume" in df.columns:
    #     volume = df["Volume"]
    # elif "tick_volume" in df.columns:
    #     volume = df["tick_volume"]
    # elif "Volume" in df.columns:
    #     volume = df["Volume"]
    # else:
    #     raise ValueError("No volume column found.")

    volume = df["range"]

    # print(type(volume))
    # print(volume.shape)
    # print(df.columns.tolist())

    return round(volume.rolling(period).mean(), 2)

def BullishBearish(df):
    """
    Returns:
        bullish (np.ndarray[bool])
        bearish (np.ndarray[bool])
    """

    open = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    bullish = close > open
    bearish = close < open

    return bullish, bearish

def add_irl_erl(df, swing_window=5):
    """
    Adds:
        range_high
        range_low
        erl_high
        erl_low
        irl

    Assumes df contains:
        High
        Low
        Close
    """

    # df = df.copy()

    # ----------------------------
    # Current dealing range
    # ----------------------------
    df["range_high"] = np.where(df["swing_high"], df["High"], np.nan)
    df["range_low"] = np.where(df["swing_low"], df["Low"], np.nan)

    df["range_high"] = df["range_high"].ffill()
    df["range_low"] = df["range_low"].ffill()

    # ----------------------------
    # External liquidity
    # ----------------------------
    df["erl_high"] = df["High"] > df["range_high"]
    df["erl_low"] = df["Low"] < df["range_low"]

    # ----------------------------
    # Internal liquidity
    # ----------------------------
    df["irl"] = (
        (df["High"] <= df["range_high"])
        & (df["Low"] >= df["range_low"])
    )

    return df

def GetKillzone(df, asia=(1, 4), london=(10, 13), newyork=(15, 18)):
    """
    Returns:
        0 = None
        1 = Asia
        2 = London
        3 = New York

    Expects df.index to be a DatetimeIndex.
    """

    hours = df.index.hour

    session = np.zeros(len(df), dtype=np.int8)

    session[(hours >= asia[0]) & (hours < asia[1])] = 1
    session[(hours >= london[0]) & (hours < london[1])] = 2
    session[(hours >= newyork[0]) & (hours < newyork[1])] = 3

    return session

def BuyScore(df):
    score = (
        # Trend filter
        (df["EMA7"] > df["EMA21"]).astype(int) +
        (df["EMA721_DIFF"] > 0).astype(int) +
        (df["EMA7_Slope"] > 0).astype(int) +
        (df["EMA2150_DIFF"] > 0).astype(int) +
        (df["EMA21_Slope"] > 0).astype(int) +
        (df["EMA50"] > df["EMA200"]).astype(int) +
        (df["EMA50200_DIFF"] > 0).astype(int) +
        (df["EMA50_Slope"] > 0).astype(int) +
        (df["EMA200_Slope"] > 0).astype(int) +
        (df["+di"] > df["-di"]).astype(int) +
        (df["adx"] > 20).astype(int) +
        (df["k"] > df["k_smooth"]).astype(int) +

        # OB mitigation
        df["bullish_ob_mitigation"] +

        # Breaker Block
        df["bullish_bb"] +

        # MSS
        (df["bullish_mss"] &
        (df["bullish_trend"] == 0)
        ).astype(int) +

        df["bullish_choch"] +

        # MSS Retest
        df["bullish_mss_retest"] +

        # FVG + MB/RB confluence
        (
            df["bullish_fvg_retracement"] &
            (df["bullish_mb"] | df["bullish_rb"])
        ).astype(int) +

        # Liquidity
        ((df["asia_high_dist"] >= 0) & (df["asia_high_dist"] <= 3)).astype(int) +
        ((df["asia_low_dist"] >= -3) & (df["asia_low_dist"] <= 0)).astype(int) +
        ((df["pdh_dist"] >= 0) & (df["pdh_dist"] <= 3)).astype(int) +
        ((df["pdl_dist"] >= -3) & (df["pdl_dist"] <= 0)).astype(int) +

        # Equal lows
        df["eql"] +

        # Fib
        df["bullish_fib382"] +
        df["bullish_fib500"] +
        df["bullish_fib618"]
    )

    score *= (df["killzone"].isin([1, 2, 3])).astype(int)
    score *= (df["adx"] >= 20).astype(int)

    return score

def SellScore(df):
    score = (
        # Trend filter
        (df["EMA7"] < df["EMA21"]).astype(int) +
        (df["EMA721_DIFF"] < 0).astype(int) +
        (df["EMA7_Slope"] < 0).astype(int) +
        (df["EMA21"] < df["EMA50"]).astype(int) +
        (df["EMA2150_DIFF"] < 0).astype(int) +
        (df["EMA21_Slope"] < 0).astype(int) +
        (df["EMA50"] < df["EMA200"]).astype(int) +
        (df["EMA50200_DIFF"] < 0).astype(int) +
        (df["EMA50_Slope"] < 0).astype(int) +
        (df["EMA200_Slope"] < 0).astype(int) +
        (df["-di"] > df["+di"]).astype(int) +
        (df["adx"] > 20).astype(int) +
        (df["k"] < df["k_smooth"]).astype(int) +

        # OB mitigation
        df["bearish_ob_mitigation"] +

        # Breaker Block
        df["bearish_bb"] +

        # MSS
        (df["bearish_mss"] &
        (df["bearish_trend"] == 0)
        ).astype(int) +

        # MSS Retest
        df["bearish_mss_retest"] +

        df["bearish_choch"] +

        # FVG + MB/RB confluence
        (
            df["bearish_fvg_retracement"] &
            (df["bearish_mb"] | df["bearish_rb"])
        ).astype(int) +

        # Liquidity
        ((df["asia_high_dist"] >= -3) & (df["asia_high_dist"] <= 0)).astype(int) +
        ((df["asia_low_dist"] >= 0) & (df["asia_low_dist"] <= 3)).astype(int) +
        ((df["pdh_dist"] >= -3) & (df["pdh_dist"] <= 0)).astype(int) +
        ((df["pdl_dist"] >= 0) & (df["pdl_dist"] <= 3)).astype(int) +

        # Equal highs
        df["eqh"] +

        # Fib
        df["bearish_fib382"] +
        df["bearish_fib500"] +
        df["bearish_fib618"]
    )

    score *= (df["killzone"].isin([1, 2, 3])).astype(int)
    score *= (df["adx"] >= 20).astype(int)

    return score

def add_indicators(df):
    # print("Loading indicators")
    df['adx'], df['+di'], df['-di'] = ADX(df)

    df['k'], df['k_smooth'] = STOCH(df)

    df['EMA7'] = EMA(df, 7)
    # df['EMA1'] = EMA(df, 1)
    # df['EMA1_Slope'] = df['EMA1'].diff()
    df['EMA7_Slope'] = df['EMA7'].diff()
    df['EMA21'] = EMA(df, 21)
    df['EMA21_Slope'] = df['EMA21'].diff()
    df['EMA721_DIFF'] = df['EMA7'] - df['EMA21']
    df['EMA50'] = EMA(df, 50)
    df['EMA2150_DIFF'] = df['EMA21'] - df['EMA50']
    df['EMA50_Slope'] = df['EMA50'].diff()
    df['EMA200'] = EMA(df, 200)
    df['EMA50200_DIFF'] = df['EMA50'] - df['EMA200']
    df['EMA200_Slope'] = df['EMA200'].diff()
    # df['EMA1'] = EMA(df, 1)
    # df['EMA71_DIFF'] = df['EMA1'] - df['EMA7']

    df["vwap"], df["vwap_upper"], df["vwap_lower"], df["above_vwap"], df["below_vwap"], df["vwap_above_upper"], df["vwap_below_lower"], df["vwap_slope"] = VWAP(df)

    df["range"] = GetRange(df)
    df["range_ma"] = RangeMA(df)

    df["indecision"] = Indecision(df)

    df["bullish_ob"] = BullishOB(df)
    df["bearish_ob"] = BearishOB(df)
    df["bullish_ob_mitigation"], df["bearish_ob_mitigation"] = OBMitigation(df)

    df["bullish_fvg"] = BullishFVG(df)
    df["bearish_fvg"] = BearishFVG(df)
    df["bullish_fvg_retracement"], df["bearish_fvg_retracement"] = FVGRetracement(df)
    df["bullish_ifvg"] = BullishIFVG(df)
    df["bearish_ifvg"] = BearishIFVG(df)

    df["bullish"], df["bearish"] = BullishBearish(df)

    df["eqh"], df["eql"] = EQH_EQL(df)
    df["eqh_retest"], df["eql_retest"] = EQHEQLRetest(df)

    df["bearish_mb"] = BearishMB(df)
    df["bullish_mb"] = BullishMB(df)

    df["bullish_rb"], df["bearish_rb"] = RejectionBlocks(df)

    df["swing_high"], df["swing_low"] = Swings(df)
    df["bullish_mss"], df["bearish_mss"] = MSS(df)
    df["bullish_mss_retest"], df["bearish_mss_retest"] = MSS_Retest(df)

    df["bullish_bos"], df["bearish_bos"] = BoS(df)
    df["bullish_trend"], df["bearish_trend"] = Trend(df)
    df["bullish_choch"], df["bearish_choch"] = CHoCH(df)

    df["bullish_bb"] = BullishBB(df)
    df["bearish_bb"] = BearishBB(df)

    df["exhaustion"] = ExhaustionCandle(df)

    df["above_support"], df["below_resistance"] = TrendLines(df)

    df["bullish_fib382"], df["bullish_fib500"], df["bullish_fib618"], df["bullish_fib786"], df["bearish_fib382"], df["bearish_fib500"], df["bearish_fib618"], df["bearish_fib786"] = Fibonacci(df)

    df = add_irl_erl(df)

    df["killzone"] = GetKillzone(df)

    """
    if "Volume" in df.columns:
        df.rename(columns={
            'Volume': 'volume',
        }, inplace=True)
    elif "tick_volume" in df.columns:
        df.rename(columns={
            'tick_volume': 'volume',
        }, inplace=True)
    """

    df["asia_high_dist"] = AsiaHighDistance(df)
    df["asia_low_dist"] = AsiaLowDistance(df)

    df["pdh_dist"] = PDHDistance(df)
    df["pdl_dist"] = PDLDistance(df)

    df["pd_poc_dist"] = PDPOCDistance(df)

    df["sell_score"] = SellScore(df)
    df["buy_score"] = BuyScore(df)

    df = df[["Open", "High", "Low", "Close",
            "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA721_DIFF", "EMA7_Slope", "EMA21_Slope", "EMA50", "EMA200", "EMA2150_DIFF", "EMA50_Slope", "EMA50200_DIFF", "EMA200_Slope",
            "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb","bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
            "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
            "swing_high", "swing_low", "bullish_mss", "bearish_mss", "bullish_mss_retest", "bearish_mss_retest",
            "bullish_bos", "bearish_bos", "bullish_trend", "bearish_trend", "bullish_choch", "bearish_choch",
            "range_high", "range_low", "erl_high", "erl_low", "irl",
            "exhaustion",
            "above_support", "below_resistance",
            "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
            "vwap", "vwap_upper", "vwap_lower", "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
            "range_ma", "range",
            "bullish", "bearish",
            "asia_high_dist", "asia_low_dist",
            "pdh_dist", "pdl_dist",
            "pd_poc_dist",
            "killzone",
            "sell_score", "buy_score",
            ]].copy()

    # print("Loaded indicators")

    df.dropna(inplace=True)
    return df

class PPOxLSTMNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        hidden_size=128,
        action_size=3,
        seq_len=32,
    ):
        super().__init__()

        self.embedding = nn.Linear(state_size, hidden_size)

        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                )
            ),

            slstm_at=[0, 1],

            context_length=seq_len,
            num_blocks=2,
            embedding_dim=hidden_size,
        )

        self.backbone = xLSTMBlockStack(cfg)

        # Buy / Sell / Hold
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        # RR multiplier
        self.rr_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # SL multiplier
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # PPO critic
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # (batch, seq, features)
        x = self.embedding(x)

        x = self.backbone(x)

        # Last timestep
        h = x[:, -1]

        # Action logits
        logits = self.policy(h)

        # RR between 1 and 10
        rr_mult = 1.0 + torch.sigmoid(
            self.rr_head(h)
        ) * 9.0

        # SL multiplier between 0.75x and 2.0x
        sl_mult = 0.75 + torch.sigmoid(
            self.sl_head(h)
        ) * 1.25

        # PPO critic
        value = self.value(h).squeeze(-1)

        return logits, rr_mult, sl_mult, value

class xLSTMPPOAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        lr=3e-4,
        gamma=0.95,
        clip_ratio=0.2,
        gae_lambda=0.95
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda

        self.train_epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = PPOxLSTMNetwork(
            state_size,
            hidden_size,
            action_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.trajectory = []

    def _state_tensor(self, state_seq):
        return torch.tensor(
            state_seq,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def select_action(self, state_seq, in_position=False, training=False):

        state = self._state_tensor(state_seq)

        with torch.no_grad():
            logits, rr_mult, sl_mult, value = self.model(state)

        logits = logits.squeeze(0)
        rr_mult = rr_mult.squeeze()
        sl_mult = sl_mult.squeeze()

        valid_actions = [0] if in_position else [0,1,2]

        masked_logits = logits.clone()

        for i in range(self.action_size):
            if i not in valid_actions:
                masked_logits[i] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample() if training else torch.argmax(probs)

        logprob = dist.log_prob(action)

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item()),
            float(rr_mult.item()),
            float(sl_mult.item())
        )

    def store_transition(
        self,
        state_seq,
        action,
        logprob,
        value,
        rr_mult,
        sl_mult,
        reward,
        done
    ):
        self.trajectory.append(
            (
                np.array(state_seq, dtype=np.float32),
                action,
                logprob,
                value,
                rr_mult,
                sl_mult,
                reward,
                done
            )
        )

    def compute_gae(self, rewards, values, dones):

        advantages = []
        gae = 0

        values = np.append(values, 0.0)

        for t in reversed(range(len(rewards))):

            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[t])
                * gae
            )

            advantages.insert(0, gae)

        return np.array(advantages, dtype=np.float32)

    def train(self):

        if len(self.trajectory) < 32:
            return

        (
            states,
            actions,
            old_logprobs,
            values,
            rr_mults,
            sl_mults,
            rewards,
            dones
        ) = zip(*self.trajectory)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n = len(states)

        for _ in range(self.train_epochs):

            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):

                batch = idx[start:start+self.batch_size]

                logits, rr_pred, sl_pred, value_pred = self.model(states[batch])

                dist = Categorical(logits=logits)

                new_logprob = dist.log_prob(actions[batch])

                ratio = torch.exp(new_logprob - old_logprobs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(
                    ratio,
                    1-self.clip_ratio,
                    1+self.clip_ratio
                ) * advantages[batch]

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(
                    value_pred,
                    returns[batch]
                )

                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        self.trajectory.clear()

    def savecheckpoint(self, symbol):
        os.makedirs("LSTM-PPO-saves", exist_ok=True)

        filename = (
            f"LSTM-PPO-saves/"
            f"{datetime.now().strftime('%Y-%m-%d')}-"
            f"{symbol}.checkpoint.pt"
        )

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            },
            filename
        )

    def loadcheckpoint(self, symbol):

        if not os.path.exists("LSTM-PPO-saves"):
            return

        files = [
            os.path.join("LSTM-PPO-saves", f)
            for f in os.listdir("LSTM-PPO-saves")
            if f.endswith(".checkpoint.pt")
            and symbol in f
        ]

        if not files:
            return

        latest = max(files, key=os.path.getmtime)

        checkpoint = torch.load(
            latest,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        print(f"Loaded checkpoint: {latest}")

class PPOLSTMNetwork(nn.Module):
    def __init__(self, state_size=12, hidden_size=64, action_size=3):
        super().__init__()

        # print("PPOLSTMNetwork state_size =", state_size)

        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # print("LSTM input_size =", self.lstm.input_size)

        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # self.debug = False

    def forward(self, x):
        """
        if self.debug:
            print("x shape before lstm:", x.shape)

        if x.shape[1] == 0:
            print("ERROR: zero sequence length")
            print("x shape:", x.shape)
            raise ValueError("Zero sequence length")
        """

        out, _ = self.lstm(x)
        h = out[:, -1, :]

        logits = self.policy(h)
        value = self.value(h).squeeze(-1)

        return logits, value

class LSTMPPOAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        lr=3e-4,
        gamma=0.95,
        clip_ratio=0.2,
        gae_lambda=0.95
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda

        self.train_epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = PPOLSTMNetwork(
            state_size,
            hidden_size,
            action_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.trajectory = []

        # print("Agent state_size =", state_size)

    def _state_tensor(self, state_seq):
        return torch.tensor(
            state_seq,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def select_action(self, state_seq, in_position=False, training=False):

        state = self._state_tensor(state_seq)

        # if training is False:
        #     print("state type:", type(state))
        #     print("state shape:", state.shape if hasattr(state, "shape") else "no shape")

        with torch.no_grad():
            logits, value = self.model(state)

        logits = logits.squeeze(0)

        if in_position:
            valid_actions = [0]
        else:
            valid_actions = [0, 1, 2]

        masked_logits = logits.clone()

        for i in range(self.action_size):
            if i not in valid_actions:
                masked_logits[i] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)

        dist = Categorical(probs)

        if training:
            action = dist.sample()
        else:
            action = torch.argmax(probs)

        logprob = dist.log_prob(action)

        """
        print(
            f"H={probs[0]:.2f} "
            f"B={probs[1]:.2f} "
            f"S={probs[2]:.2f}"
        )
        """

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item())
        )

    def store_transition(
        self,
        state_seq,
        action,
        logprob,
        value,
        reward,
        done
    ):
        self.trajectory.append(
            (
                np.array(state_seq, dtype=np.float32),
                action,
                logprob,
                value,
                reward,
                done
            )
        )

    def compute_gae(self, rewards, values, dones):

        advantages = []
        gae = 0

        values = np.append(values, 0.0)

        for t in reversed(range(len(rewards))):

            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[t])
                * gae
            )

            advantages.insert(0, gae)

        return np.array(advantages, dtype=np.float32)

    def train(self):

        if len(self.trajectory) < 32:
            return

        states, actions, old_logprobs, values, rewards, dones = zip(
            *self.trajectory
        )

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        old_logprobs = np.array(old_logprobs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        advantages = self.compute_gae(
            rewards,
            values,
            dones
        )

        returns = advantages + values

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        old_logprobs = torch.tensor(
            old_logprobs,
            dtype=torch.float32,
            device=self.device
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )

        n = len(states)

        for _ in range(self.train_epochs):

            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):

                batch_idx = idx[start:start+self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                # print("b_states.shape =", b_states.shape)
                # print("model input_size =", self.model.lstm.input_size)
                logits, values_pred = self.model(b_states)

                dist = Categorical(logits=logits)

                new_logprobs = dist.log_prob(
                    b_actions
                )

                entropy = dist.entropy().mean()

                ratio = torch.exp(
                    new_logprobs - b_old_logprobs
                )

                surr1 = ratio * b_advantages

                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_ratio,
                    1 + self.clip_ratio
                ) * b_advantages

                policy_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                value_loss = F.mse_loss(
                    values_pred,
                    b_returns
                )

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

        self.trajectory.clear()

    def savecheckpoint(self, symbol):

        os.makedirs(
            "LSTM-PPO-saves",
            exist_ok=True
        )

        filename = (
            f"LSTM-PPO-saves/"
            f"{datetime.now().strftime('%Y-%m-%d')}-"
            f"{symbol}.checkpoint.pt"
        )

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            },
            filename
        )

    def loadcheckpoint(self, symbol):

        if not os.path.exists("LSTM-PPO-saves"):
            return

        files = [
            os.path.join("LSTM-PPO-saves", f)
            for f in os.listdir("LSTM-PPO-saves")
            if f.endswith(".checkpoint.pt")
            and symbol in f
        ]

        if not files:
            return

        latest = max(files, key=os.path.getmtime)

        checkpoint = torch.load(
            latest,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        print(f"Loaded checkpoint: {latest}")

class WinRateKNN:
    def __init__(self, symbol, k=10):
        self.k = k
        self.symbol = symbol
        self.states = []
        self.labels = []  # 1 = win, 0 = loss
        self.model = None

    def add(self, state, is_win):
        try:
            state = np.array(state, dtype=np.float32).flatten()  # Force all elements to float
        except Exception as e:
            # print("❌ Could not convert state to float:", state, "| Error:", e)
            return

        if not np.all(np.isfinite(state)):
            # print("⚠️ Skipping state with NaN or Inf:", state)
            return

        self.states.append(state)
        self.labels.append(1 if is_win else 0)

        if len(self.states) >= 100:
            self._remove_redundant_neighbor()
            # self.states.pop(0)
            # self.labels.pop(0)

        if len(self.states) >= self.k:
            self._fit()

    def _remove_redundant_neighbor(self):
        if len(self.states) < 2:
            return  # Nothing to remove

        X = np.array(self.states)

        # Compute pairwise similarity (cosine, or use euclidean if you prefer)
        sim_matrix = cosine_similarity(X)

        # Zero out diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, 0)

        # Compute average similarity for each row (how redundant each entry is)
        redundancy_scores = sim_matrix.mean(axis=1)

        # Remove the most redundant (highest avg similarity)
        idx_to_remove = np.argmax(redundancy_scores)

        del self.states[idx_to_remove]
        del self.labels[idx_to_remove]
    def _fit(self):
        """
        Fit the KNN model with stored data.
        """
        if len(self.states) < 1:
            # print("⚠️ Not enough data to fit KNN.")
            return

        # Safety check
        k_neighbors = max(1, min(self.k, len(self.states)))

        self.model = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree")
        self.model.fit(self.states)

    def predict_win_rate(self, state_seq, k_near=5, k_far=5):
        """
        Return the win rate based on k nearest neighbors of the input state.
        """
        # if not self.model or len(self.states) < self.k:
        if len(self.states) < 1000:
            # return True  # Not enough data
            return 1  # Not enough data

        # Find the 100 nearest neighbors
        distances, indices = self.model.kneighbors(state.reshape(1, -1), n_neighbors=50)
        distances = distances[0]
        indices = indices[0]

        # Split into nearest and farthest groups
        nearest_idx = indices[:k_near]
        nearest_dist = distances[:k_near]

        farthest_idx = indices[-k_far:]
        farthest_dist = distances[-k_far:]

        # Combine indices and distances
        combined_idx = np.concatenate([nearest_idx, farthest_idx])
        combined_dist = np.concatenate([nearest_dist, farthest_dist])

        # Get win/loss labels for selected neighbors
        selected_labels = np.array([self.labels[i] for i in combined_idx])

        # Calculate weights (closer gets higher weight)
        weights = 1 / (combined_dist + 1e-6)  # Add epsilon to avoid div-by-zero

        # Normalize weights
        weights /= weights.sum()

        # Compute weighted win rate
        win_rate = np.dot(selected_labels, weights)

        return win_rate
    def save(self):
        """
        Save the KNN model to disk.
        """
        path = f"LSTM-PPO-saves/{datetime.now().strftime('%Y-%m-%d')}-{self.symbol}.win_rate_knn.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "states": self.states,
                "labels": self.labels,
                "model": self.model
            }, f)

    def load(self):
        """
        Load the KNN model from disk.
        """
        # path = f"LSTM-PPO-saves/win_rate_knn-{self.symbol}.pkl"
        files = sorted(os.listdir("LSTM-PPO-saves"))
        files = [f for f in files if f.endswith(".win_rate_knn.pkl") and self.symbol in f]
        if not files:
            print(f"[!] No checkpoint found for {self.symbol}")
            return

        latest = os.path.join("LSTM-PPO-saves", files[-1])

        try:
            with open(latest, "rb") as f:
                data = pickle.load(f)
                self.states = data["states"]
                self.labels = data["labels"]
                self.model = data["model"]
                # print(f"✅ Loaded WinRateKNN from {latest}")
        except FileNotFoundError:
            print(f"⚠️ No saved KNN found at {path}. Starting fresh.")
    
def sharpe_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return (mean_ret - risk_free_rate) / std_ret

def sortino_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    # Downside deviation: only consider returns below risk-free rate, and their square differences
    downside_diff = [(r - risk_free_rate)**2 for r in returns if r < risk_free_rate]
    
    if len(downside_diff) == 0:
        return 0  # Or float('inf') if you'd rather signal perfect performance
    
    downside_std = np.sqrt(np.mean(downside_diff))
    
    if downside_std == 0:
        return 0
    
    return (mean_ret - risk_free_rate) / downside_std

def rolling_sharpe(trade_returns_50):
    if len(trade_returns_50) < 10:
        return 0.0

    r = np.asarray(trade_returns_50, dtype=np.float32)

    std = r.std()
    if std < 1e-8:
        return 0.0

    return r.mean() / std

def max_drawdown(returns):

    if len(returns) == 0:
        return 0

    equity = np.cumsum(returns)

    peak = equity[0]
    max_dd = 0

    for value in equity:

        peak = max(peak, value)

        dd = peak - value

        max_dd = max(max_dd, dd)

    return max_dd

def SelectAction(df, i):

    # Session filter
    if df["killzone"].iloc[i] not in (1, 2, 3):   # Asia, London, NY
        return 0

    # Trend strength
    if df["adx"].iloc[i] < 20:
        return 0

    buy = 0
    sell = 0

    # -------------------------
    # OB mitigation
    # -------------------------
    buy += int(df["bullish_ob_mitigation"].iloc[i])
    sell += int(df["bearish_ob_mitigation"].iloc[i])

    # Breaker block
    buy += int(df["bullish_bb"].iloc[i])
    sell += int(df["bearish_bb"].iloc[i])

    # MSS
    buy += int(
        df["bullish_mss"].iloc[i] and
        not df["bullish_trend"].iloc[i]
    )

    sell += int(
        df["bearish_mss"].iloc[i] and
        not df["bearish_trend"].iloc[i]
    )

    # MSS Retest
    buy += int(df["bullish_mss_retest"].iloc[i])
    sell += int(df["bearish_mss_retest"].iloc[i])

    # FVG + MB/RB
    buy += int(
        df["bullish_fvg_retracement"].iloc[i]
        and (df["bullish_mb"].iloc[i] or df["bullish_rb"].iloc[i])
    )

    sell += int(
        df["bearish_fvg_retracement"].iloc[i]
        and (df["bearish_mb"].iloc[i] or df["bearish_rb"].iloc[i])
    )

    # Asia High
    buy += int(0 <= df["asia_high_dist"].iloc[i] <= 3)
    sell += int(-3 <= df["asia_high_dist"].iloc[i] <= 0)

    # Asia Low
    buy += int(-3 <= df["asia_low_dist"].iloc[i] <= 0)
    sell += int(0 <= df["asia_low_dist"].iloc[i] <= 3)

    # PDH
    buy += int(0 <= df["pdh_dist"].iloc[i] <= 3)
    sell += int(-3 <= df["pdh_dist"].iloc[i] <= 0)

    # PDL
    buy += int(-3 <= df["pdl_dist"].iloc[i] <= 0)
    sell += int(0 <= df["pdl_dist"].iloc[i] <= 3)

    # Equal High / Equal Low
    buy += int(df["eql"].iloc[i])
    sell += int(df["eqh"].iloc[i])

    # 0.382, 0.5, 0.618 retracement
    buy += int(df["bullish_fib382"].iloc[i])
    sell += int(df["bearish_fib382"].iloc[i])
    buy += int(df["bullish_fib500"].iloc[i])
    sell += int(df["bearish_fib500"].iloc[i])
    buy += int(df["bullish_fib618"].iloc[i])
    sell += int(df["bearish_fib618"].iloc[i])

    # Final decision
    if buy >= 2 and buy > sell:
        return 1

    if sell >= 2 and sell > buy:
        return 2

    return 0

def train_bot(symbol="XAUUSD"):

    print("Training bot")
    start = time.perf_counter()

    df = load_last_mb_xauusd()

    print(f"Loading indicators... ({time.strftime('%H:%M')})")

    df = add_indicators(df)

    start = time.perf_counter()
    # elapsed = int((time.perf_counter() - start) // 60)
    elapsed = int((time.perf_counter() - start))
    print(f"Loaded indicators (Elapsed: {elapsed}m)")
    print(f"Loading checkpoint....")


    SEQ_LEN = 12 * 3

    save_count = 1440

    FEATURES = [
        "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA721_DIFF", "EMA7_Slope", "EMA21_Slope", "EMA50", "EMA200", "EMA2150_DIFF", "EMA50_Slope", "EMA50200_DIFF", "EMA200_Slope",
        "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb","bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
        "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
        "swing_high", "swing_low", "bullish_mss", "bearish_mss", "bullish_mss_retest", "bearish_mss_retest",
        "bullish_bos", "bearish_bos", "bullish_trend", "bearish_trend", "bullish_choch", "bearish_choch",
        "range_high", "range_low", "erl_high", "erl_low", "irl",
        "exhaustion",
        "above_support", "below_resistance",
        "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
        "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
        "range_ma", "range",
        "bullish", "bearish",
        "asia_high_dist", "asia_low_dist",
        "pdh_dist", "pdl_dist",
        "pd_poc_dist",
        "killzone",
        "sell_score", "buy_score"
    ]

    agent = xLSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=3
    )

    agent.loadcheckpoint(symbol)
    print(f"Loaded checkpoint... ({time.strftime('%H:%M')})")

    save_counter = 0

    in_position = False
    position_type = None

    entry_price = 0
    sl_price = 0
    tp1_price = 0.0
    tp2_price = 0.0

    trade1_open = False
    trade2_open = False

    be_moved = False

    trade_returns = []

    rr_list = []
    SPREAD_AND_COMMISSION = 6

    sl_pips = 0
    sl_list = []

    PIP_VALUE = 0.1

    training_start_2 = time.time()
    training_start_3 = time.time()

    state_buffer = []

    for i in range(len(df)):
        row = df.iloc[i][FEATURES].values.astype(np.float32)
        state_buffer.append(row)

    # ------------------------------------------------------------------
    # Week / month / recovery-factor training config
    # ------------------------------------------------------------------
    WEEK_SIZE = save_count          # 1440 M5 candles ≈ 1 trading week
    WEEKS_PER_MONTH = 4             # agent.trajectory is reset every 4 weeks
    RF_TARGET = 1.5                 # keep retraining a week until RF exceeds this
    MAX_ATTEMPTS_PER_WEEK = 20      # safety cap so a "dead" week can't hang forever
    LOOP_FOREVER = True             # True: wrap back to the start of df and repeat
                                     # False: stop once len(df) has been covered

    total_weeks = max(1, math.ceil((len(df) - SEQ_LEN) / WEEK_SIZE))

    week_idx = 0
    week_start = SEQ_LEN

    while True:

        # Wrap back to the beginning of df once we've covered it all
        if week_start >= len(df):
            if not LOOP_FOREVER:
                break
            week_start = SEQ_LEN
            week_idx = 0

        week_end = min(week_start + WEEK_SIZE, len(df))

        # Reset the trajectory at the start of every new "month"
        if week_idx % WEEKS_PER_MONTH == 0:
            agent.trajectory.clear()

        rf = 0.0
        attempt = 0

        # ==============================================================
        # Keep retraining THIS week until its recovery factor beats the
        # target, then fall through and move on to the next week.
        # ==============================================================
        while rf <= RF_TARGET and attempt < MAX_ATTEMPTS_PER_WEEK:

            attempt += 1

            # Per-attempt trading state resets (replaying the same week)
            in_position = False
            position_type = None
            entry_price = 0
            sl_price = 0
            tp1_price = 0.0
            tp2_price = 0.0
            trade1_open = False
            trade2_open = False
            be_moved = False

            trade_returns = []
            rr_list = []
            sl_list = []
            sl_pips = 0

            pending_entry = None   # caches the decision that opened the current trade

            training_start_3 = time.time()

            for i in range(week_start, week_end):

                current = df.iloc[i]

                current_price = current["Close"]
                high = current["High"]
                low = current["Low"]
                base_sl = 50

                state_seq = np.array(df.iloc[i - SEQ_LEN:i][FEATURES], dtype=np.float32)

                result = agent.select_action(state_seq, in_position, training=True)

                if result is None:
                    continue

                action, logprob, value, rr_mult, sl_mult = result
                rr_list.append(rr_mult)

                pnl = 0.0
                done = False

                if in_position:
                    action = 0

                # ==========================================================
                # OPEN LONG
                # ==========================================================
                if action == 1:

                    in_position = True
                    trade1_open = True
                    trade2_open = True
                    position_type = "long"
                    be_moved = False

                    entry_price = current_price

                    sl_pips = np.clip(base_sl * sl_mult, 20, 50)
                    sl_list.append(sl_pips)

                    sl_price = entry_price - sl_pips * PIP_VALUE
                    full_tp_distance = sl_pips * rr_mult * PIP_VALUE

                    tp1_price = entry_price + full_tp_distance * 0.5
                    tp2_price = entry_price + full_tp_distance

                    # Remember what produced this trade so the eventual
                    # outcome is credited to THIS decision, not to a later
                    # unrelated one.
                    pending_entry = dict(
                        state_seq=state_seq,
                        action=action,
                        logprob=logprob,
                        value=value,
                        rr_mult=rr_mult,
                        sl_mult=sl_mult,
                    )

                # ==========================================================
                # OPEN SHORT
                # ==========================================================
                elif action == 2:

                    in_position = True
                    trade1_open = True
                    trade2_open = True
                    position_type = "short"
                    be_moved = False

                    entry_price = current_price

                    sl_pips = np.clip(base_sl * sl_mult, 20, 50)
                    sl_list.append(sl_pips)

                    sl_price = entry_price + sl_pips * PIP_VALUE
                    full_tp_distance = sl_pips * rr_mult * PIP_VALUE

                    tp1_price = entry_price - full_tp_distance * 0.5
                    tp2_price = entry_price - full_tp_distance

                    pending_entry = dict(
                        state_seq=state_seq,
                        action=action,
                        logprob=logprob,
                        value=value,
                        rr_mult=rr_mult,
                        sl_mult=sl_mult,
                    )

                # ==========================================================
                # MANAGE POSITION
                # ==========================================================
                if in_position:

                    trade_closed = False

                    if position_type == "long":

                        if trade1_open and high >= tp1_price:
                            pnl += sl_pips * rr_mult * 0.5 - SPREAD_AND_COMMISSION
                            trade1_open = False
                            if not be_moved:
                                sl_price = entry_price
                                be_moved = True

                        if trade2_open and high >= tp2_price:
                            pnl += sl_pips * rr_mult - SPREAD_AND_COMMISSION
                            trade2_open = False
                            trade_closed = True

                        if not trade_closed and low <= sl_price:
                            if sl_price != entry_price:
                                pnl -= sl_pips * 2 + SPREAD_AND_COMMISSION * 2
                            trade1_open = False
                            trade2_open = False
                            trade_closed = True

                    elif position_type == "short":

                        if trade1_open and low <= tp1_price:
                            pnl += sl_pips * rr_mult * 0.5 - SPREAD_AND_COMMISSION
                            trade1_open = False
                            if not be_moved:
                                sl_price = entry_price
                                be_moved = True

                        if trade2_open and low <= tp2_price:
                            pnl += sl_pips * rr_mult - SPREAD_AND_COMMISSION
                            trade2_open = False
                            trade_closed = True

                        if not trade_closed and high >= sl_price:
                            if sl_price != entry_price:
                                pnl -= sl_pips * 2 + SPREAD_AND_COMMISSION * 2
                            trade1_open = False
                            trade2_open = False
                            trade_closed = True

                    # Any close — win via TP2 or loss via SL — ends the episode.
                    if trade_closed:
                        in_position = False
                        done = True
                        trade_returns.append(pnl)

                        if pending_entry is not None:
                            reward = pnl / 50.0  # ~R-multiple, stable scale

                            agent.store_transition(
                                pending_entry["state_seq"],
                                pending_entry["action"],
                                pending_entry["logprob"],
                                pending_entry["value"],
                                rr_mult=pending_entry["rr_mult"],
                                sl_mult=pending_entry["sl_mult"],
                                reward=reward,
                                done=True,
                            )
                            pending_entry = None

                        pnl = 0

            # ==========================================================
            # END OF THIS PASS OVER THE WEEK: compute stats + RF
            # ==========================================================

            if len(trade_returns) > 5:

                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]

                weekly_pnl = np.sum(trade_returns)

                winrate = (
                    len(wins) / len(trade_returns)
                    if len(trade_returns) > 0 else 0
                )

                mean_win = (
                    np.mean(wins)
                    if len(wins) > 0 else 0
                )

                mean_loss = (
                    np.mean(losses)
                    if len(losses) > 0 else 0
                )

                sharpe = sharpe_ratio(trade_returns)
                sortino = sortino_ratio(trade_returns)

                gross_profit = sum(wins)
                gross_loss = abs(sum(losses))
                profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0
                    else float("inf")
                )
                avg_rr = sum(rr_list) / len(rr_list) if rr_list else 0
                avg_sl = sum(sl_list) / len(sl_list) if sl_list else 0
                max_dd = max_drawdown(trade_returns)
                R_pnl = weekly_pnl / avg_sl if avg_sl else 0

                if max_dd > 0 and avg_sl > 0:
                    rf = R_pnl / (max_dd / ((avg_sl / 10) * 2))
                else:
                    rf = 0.0

                print()
                print("================================================")
                print(f"[{symbol}] WEEK {week_idx + 1}/{total_weeks} "
                      f"(attempt {attempt}) PPO TRAINING")
                print("================================================")
                print(f"Trades:          {len(trade_returns)}")
                print(f"Weekly PnL:      {weekly_pnl:.0f} pips")
                print(f"Winrate:         {winrate*100:.2f}%")
                print(f"Mean Win:        {mean_win:.0f} pips")
                print(f"Mean Loss:       {mean_loss:.0f} pips")
                if avg_sl:
                    print(f"Max DD:          {max_dd/(avg_sl*2):.2f}R")
                print(f"PF:              {profit_factor:.2f}")
                print(f"RF:              {rf:.2f}")
                if avg_sl:
                    print(f"Weekly R PnL:    {R_pnl/2:.2f}R")
                print(f"Sharpe:          {sharpe:.2f}")
                print(f"Sortino:         {sortino:.2f}")
                print("================================================")
                print(
                    f"[{symbol}] [INFO] Trained on week "
                    f"(Elapsed: {timedelta(seconds=int(time.time() - training_start_3))})"
                )
                print()

            else:
                # Not enough trades to trust the RF number — force a retry
                rf = 0.0
                print(
                    f"[{symbol}] Week {week_idx + 1}/{total_weeks} "
                    f"attempt {attempt}: only {len(trade_returns)} trades, retrying..."
                )

            training_start = time.time()
            print(f"[{symbol}] [INFO] Training PPO on week {week_idx + 1} "
                  f"(attempt {attempt})...")
            agent.train()
            print(
                f"[{symbol}] "
                f"[INFO] Finished training PPO "
                f"(Elapsed: {timedelta(seconds=int(time.time() - training_start))})"
            )
            agent.savecheckpoint(symbol)

            if attempt >= MAX_ATTEMPTS_PER_WEEK:
                print(
                    f"[{symbol}] [WARN] Week {week_idx + 1} hit "
                    f"MAX_ATTEMPTS_PER_WEEK ({MAX_ATTEMPTS_PER_WEEK}) "
                    f"without RF > {RF_TARGET} (last RF={rf:.2f}); moving on anyway."
                )

        elapsed_total = time.time() - training_start_2
        print(
            f"[{symbol}] [INFO] Week {week_idx + 1}/{total_weeks} done "
            f"(RF={rf:.2f} > {RF_TARGET} after {attempt} attempt(s)) | "
            f"Total elapsed: {timedelta(seconds=int(elapsed_total))}"
        )

        # Advance to the next week
        week_start = week_end
        week_idx += 1

    agent.train()
    agent.savecheckpoint(symbol)

    print(
        f"[{symbol}] "
        f"[INFO] Finished training "
    )

    return agent

def open_long(symbol, lot_size, sl_pips, rr_mult):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.ask

    sl = entry - sl_pips / 10

    tp1 = entry + sl_pips / 10 * rr_mult

    tps = [tp1]

    for tp in tps:

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Project Helios",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode}")
            return None

        # return result.position

        # print(result)

def open_short(symbol, lot_size, sl_pips, rr_mult):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.bid

    sl = entry + sl_pips / 10

    tp1 = entry - sl_pips / 10 * rr_mult,

    tps = [tp1]

    for tp in tps:

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Project Helios",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode}")
            return None

        # return result.position

        # print(result)

def open_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    positions = [
        p
        for p in positions
        if p.magic == 123456
    ]
    return len(positions)

def close_trades(magic=123456):
    positions = mt5.positions_get()

    if positions is None:
        print("Failed to get positions:", mt5.last_error())
        return

    for pos in positions:
        if pos.magic != magic:
            continue

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": "End of Day",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close {pos.ticket}: {result.retcode}")
        else:
            print(f"Closed {pos.ticket}")

def modify_sl():

    positions = mt5.positions_get()

    if positions is None:
        return False

    # Only our EA's positions
    positions = [p for p in positions if p.magic == 123456]

    # Only modify if exactly one position remains
    if len(positions) != 1:
        return False

    position = positions[0]

    # Already at breakeven?
    if abs(position.sl - position.price_open) < 0.01:
        return True

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": position.price_open,
        "tp": position.tp,
    }

    result = mt5.order_send(request)

    return result.retcode == mt5.TRADE_RETCODE_DONE

def test_bot(symbol="XAUUSD"):
    SEQ_LEN = 12 * 3
    
    mt5.initialize()
    account = mt5.account_info()
    balance = account.balance
    RISK = 0.0005

    FEATURES = [
        # "Open", "High", "Low", "Close",
        "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA721_DIFF", "EMA7_Slope", "EMA21_Slope", "EMA50", "EMA200", "EMA2150_DIFF", "EMA50_Slope", "EMA50200_DIFF", "EMA200_Slope",
        "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb","bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
        "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
        "swing_high", "swing_low", "bullish_mss", "bearish_mss", "bullish_mss_retest", "bearish_mss_retest",
        "bullish_bos", "bearish_bos", "bullish_trend", "bearish_trend", "bullish_choch", "bearish_choch",
        "range_high", "range_low", "erl_high", "erl_low", "irl",
        "exhaustion",
        "above_support", "below_resistance",
        "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
        #"vwap", "vwap_upper", "vwap_lower",
        "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
        "range_ma", "range",
        "bullish", "bearish",
        "asia_high_dist", "asia_low_dist",
        "pdh_dist", "pdl_dist",
        "pd_poc_dist",
        "killzone",
        "sell_score", "buy_score"
    ]

    # last_m15 = None
    last_m5 = None

    agent = xLSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=3
    )

    # agent.model.debug = True
    # try:
    agent.loadcheckpoint("XAUUSD")
    # except:
    #     print("No file for prior training, cancelling test.")
    #     return

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================

    rates_m5 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,
        600
    )

    df = pd.DataFrame(rates_m5)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'time': 'Date',
        "tick_volume": "Volume"
    }, inplace=True)
    # print(f"columns: {df.columns}")

    # df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df["Date"] = (
    pd.to_datetime(df["Date"], unit="s", utc=True)
      # .dt.tz_convert("Europe/Helsinki")
    )
    df.set_index("Date", inplace=True)

    raw_df = df

    df = add_indicators(df)

    last_m5 = df.index[-1]

    # trade = []

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    # print("entering main loop in test function")
    while True:

        # print("in main loop")
        now = datetime.now()
        position_multiplier = 1

        seconds_until_next_5m = (
            (5 - now.minute % 5) * 60
            - now.second
            - now.microsecond / 1_000_000
        )
        # print(f"sleeping {seconds_until_next_5m:.0f} seconds, current time: {datetime.now()}")
        if seconds_until_next_5m <= 0:
            seconds_until_next_5m += 30

        time.sleep(seconds_until_next_5m)
        # print(f"slept {seconds_until_next_5m:.0f} seconds, current time: {datetime.now()}")

        tick = mt5.symbol_info_tick(symbol)
        # SL_PIPS = round(tick.bid * 0.00125 * 10, 0)
        # SL_PIPS = 40

        # ======================================================
        # CHECK FOR NEW M15 CANDLE
        # ======================================================

        new_m5 = mt5.copy_rates_from_pos(
            symbol,
            mt5.TIMEFRAME_M5,
            0,
            1
        )

        current_m5 = new_m5[0]["time"]

        while True:
            if current_m5 != last_m5:
                break
            else:
                time.sleep(0.25)
                new_m5 = mt5.copy_rates_from_pos(
                    symbol,
                    mt5.TIMEFRAME_M5,
                    0,
                    1
                )
                current_m5 = new_m5[0]["time"]

        # current_m5 = new_m5[0]["time"]

        if current_m5 != last_m5:

            last_m5 = current_m5

            # ==================================================
            # APPEND NEW CANDLE
            # ==================================================

            new_row = pd.DataFrame(new_m5)

            new_row.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'time': 'Date',
                "tick_volume": "Volume"
            }, inplace=True)

            new_row["Date"] = (
                pd.to_datetime(new_row["Date"], unit="s", utc=True)
                # .dt.tz_convert("Europe/Helsinki")
            )
            new_row.set_index("Date", inplace=True)

            if new_row.index[-1] != df.index[-1]:

                raw_df = pd.concat(
                    [raw_df, new_row]
                )

                raw_df = raw_df.tail(600)

                df = add_indicators(raw_df.copy())

            # ==================================================
            # BUILD STATE SEQUENCE
            # ==================================================

            state_seq = (
                df[FEATURES]
                .tail(SEQ_LEN)
                .values
                .astype(np.float32)
            )

            # ==================================================
            # POSITION CHECK
            # ==================================================

            open_pos = open_positions(symbol)

            # ==================================================
            # PPO DECISION
            # ==================================================

            if state_seq.shape[0] != SEQ_LEN:
                print(f"Bad state shape: {state_seq.shape}")
                continue

            result = agent.select_action(state_seq, open_pos > 0, training=True)

            if result is None:
                continue

            action, logprob, value, rr_mult, sl_mult = result

            # if value < 0:
            #     value = value * -1

            current_time = df.index[-1]  # or however you store timestamps
            # print(f"time: {current_time.hour}h{current_time.minute}m")

            base_sl = max(10, df["Volume"].iloc[-1] / 100)

            if current_time.hour == 23 and current_time.minute >= 54:
                # Force close any open position
                if open_pos != 0:
                    close_trades()

                # Force HOLD
                action = 0

            # print(f"adx: {df['adx'].iloc[-1]}")
            # print(f"+di: {df['+di'].iloc[-1]}")
            # print(f"-di: {df['-di'].iloc[-1]}")
            # print(f"stoch k: {df['k'].iloc[-1]}")
            # print(f"stock d: {df['k_smooth'].iloc[-1]}")

            # if df["adx"].iloc[-1] < 20:
            #     action = 0

            # print(f"Test action: {action}")

            # ==================================================
            # OPEN NEW TRADE
            # ==================================================

            if open_pos == 0:

                # trade = None

                account = mt5.account_info()

                balance = account.balance

                # risk_per_position = max(
                #     balance * RISK / 500 / 4,
                #     0.01
                # )

                # if action == 1 and df["adx"].iloc[-1] > 20 and df["+di"].iloc[-1] > df["-di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] > 0 and df["k"].iloc[-1] < 80:
                # if action == 1 and df["buy_score"].iloc[-1] > df["sell_score"].iloc[-1]:
                # if action == 1 and df["EMA7"].iloc[-1] > df["EMA21"].iloc[-1] and df["k"].iloc[-1] < 80:
                if action == 1:
                    """
                    if df["above_vwap"].iloc[-1] == 1:
                        position_multiplier = 2
                    if df["vwap_above_upper"].iloc[-1] == 1:
                        position_multiplier = 3
                    """
                    # print(
                    #     f"[{symbol}] PPO BUY"
                    # )

                    sl_pips = np.clip(
                        base_sl * sl_mult,
                        20,
                        50
                    )

                    risk_per_position = min(
                        max((balance * RISK) / (sl_pips* 10), 0.01),
                        100.0
                    )
                    risk_per_position = round(risk_per_position, 2)

                    # sl_price = entry_price - (sl_pips * PIP_VALUE)
                    # if value < 0:
                    #     value = value * -1
                    # tp_price = entry_price + (
                    #     base_sl * rr_mult * PIP_VALUE
                    # # 2
                    # )

                    half_lot = max(round(risk_per_position / 2, 2), 0.01)

                    open_long(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult * 0.5
                    )

                    open_long(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult
                    )

                    # trade.append({
                    #     "tp1_ticket": ticket1,
                    #     "tp2_ticket": ticket2,
                    #     "entry": df["Close"].iloc[-1],
                    #     "sl": df["Close"].iloc[-1] - sl_pips / 10,
                    #     "breakeven": False,
                    # })

                # elif action == 2 and df["adx"].iloc[-1] > 20 and df["-di"].iloc[-1] > df["+di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] < 0 and df["k"].iloc[-1] > 20:
                # elif action == 2 and df["buy_score"].iloc[-1] < df["sell_score"].iloc[-1]:
                # elif action == 2 and df["EMA7"].iloc[-1] < df["EMA21"].iloc[-1] and df["k"].iloc[-1] > 20:
                elif action == 2:
                    """
                    if df["below_vwap"].iloc[-1] == 1:
                        position_multiplier = 2
                    if df["vwap_below_lower"].iloc[-1] == 1:
                        position_multiplier = 3
                    """
                    # print(
                    #     f"[{symbol}] PPO SELL"
                    # )

                    sl_pips = np.clip(
                        base_sl * sl_mult,
                        20,
                        50
                    )

                    risk_per_position = min(
                        max((balance * RISK) / (sl_pips* 10), 0.01),
                        100.0
                    )
                    risk_per_position = round(risk_per_position, 2)

                    # sl_price = entry_price + (sl_pips * PIP_VALUE)
                    # if value < 0:
                    #     value = value * -1
                    # tp_price = entry_price - (
                    #     base_sl * rr_mult * PIP_VALUE
                    #     # 2
                    # )

                    half_lot = max(round(risk_per_position / 2, 2), 0.01)

                    open_short(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult * 0.5
                    )

                    open_short(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult
                    )

                    # trade.append({
                    #     "tp1_ticket": ticket1,
                    #     "tp2_ticket": ticket2,
                    #     "entry": df["Close"].iloc[-1],
                    #     "sl": df["Close"].iloc[-1] + sl_pips / 10,
                    #     "breakeven": False,
                    # })

                # else:

                #     print(
                #         f"[{symbol}] PPO HOLD"
                #     )
            else:
                positions = mt5.positions_get()

                if positions is not None:

                    # Only this EA's positions
                    positions = [p for p in positions if p.magic == 123456]

                    # TP1 has closed, only TP2 remains
                    if len(positions) == 1:
                        modify_sl()

def get_last_date():

    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(
        CSV_FILE,
        sep=";"
    )

    if df.empty:
        return None

    return pd.to_datetime(
        df["Date"].iloc[-1]
    )

def download_xauusd_data():

    last_date = get_last_date()

    if (
        last_date is not None
        and (
            datetime.now().date()
            - last_date.date()
        ).days <= 90
    ):

        print(
            "Data already up to date."
        )

        return None

    if last_date is None:

        start_date = (
            datetime.now()
            - timedelta(days=365 * 3)
        ).strftime(
            "%Y-%m-%d"
        )

    else:

        start_date = (
            last_date
            - timedelta(days=1)
        ).strftime(
            "%Y-%m-%d"
        )

    end_date = (
        datetime.now()
        - timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"Downloading "
        f"{start_date} -> {end_date}"
    )

    subprocess.run(
        [
            # "npx",
            "dukascopy-node",
            "-i",
            "xauusd",
            "-from",
            start_date,
            "-to",
            end_date,
            "-t",
            "m5",
            "-f",
            "csv"
        ],
        check=True
    )

    files = [
        f
        for f in os.listdir(".")
        if f.startswith("xauusd")
        and f.endswith(".csv")
    ]

    if not files:

        raise FileNotFoundError(
            "No Dukascopy CSV was downloaded."
        )

    return max(
        files,
        key=os.path.getmtime
    )

def append_xauusd_data(downloaded_file):

    if downloaded_file is None:
        return

    new_df = pd.read_csv(
        downloaded_file
    )

    new_df.rename(
        columns={
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        },
        inplace=True
    )

    if os.path.exists(CSV_FILE):

        old_df = pd.read_csv(
            CSV_FILE,
            sep=";"
        )

        df = pd.concat(
            [
                old_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        df = new_df

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.to_csv(
        CSV_FILE,
        sep=";",
        index=False
    )

    os.remove(
        downloaded_file
    )

    print(
        f"Saved "
        f"{len(df)} candles "
        f"to {CSV_FILE}"
    )

def update_xauusd_data():

    downloaded_file = (
        download_xauusd_data()
    )

    append_xauusd_data(
        downloaded_file
    )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--symbol", default="XAUUSD-VIP")

    args = parser.parse_args()

    threads = []

    if args.train:
        t = threading.Thread(
            target=train_bot,
            # args=(args.symbol),
            daemon=True
        )
        t.start()
        threads.append(t)

    if args.test:
        t = threading.Thread(
            target=test_bot,
            args=(args.symbol,),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

main()
