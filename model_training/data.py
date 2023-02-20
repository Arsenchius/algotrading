import pandas as pd
import numpy as np
from binance import Client
import ta
import sys
import tqdm
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(SCRIPT_DIR)
# print(Path(os.path.dirname(SCRIPT_DIR)))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), 'backtesting'))
# print(sys.path)
from backtest import *
from strategies import *
from config import *


FEATURES = ['Open',
    'High',
    'Low',
    'Volume',
    'RSI_2',
    'SMA_20',
    'RSI_4',
    'SMA_40',
    'RSI_6',
    'SMA_60',
    'RSI_8',
    'SMA_80']
TARGET = 'Target'


def get_data(symbol, time_frame=config.INTERVAL_1MINUTE, end_date="1000 hours ago UTC"):
    client = Client()
    df = pd.DataFrame(client.get_historical_klines(symbol, time_frame, end_date))

    df = df.iloc[:, 0:6]
    df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Time", inplace=True)
    df.index = pd.to_datetime(df.index, unit="ms")
    df = df.astype(float)
    return df


def get_features():
    pass

