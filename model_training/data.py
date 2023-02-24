import pandas as pd
import numpy as np
from binance import Client
import ta
import sys
from tqdm import tqdm
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "backtesting"))
from typing import NoReturn
from backtest import *
from strategies import *
from config import *


FEATURES = ['Open',
 'High',
 'Low',
 'Close',
 'Volume',
 'RSI_14',
 'SMA_3',
 'EMA_3',
 'SMA_5',
 'EMA_5',
 'SMA_15',
 'EMA_15',
 'SMA_30',
 'EMA_30',
 'SMA_50',
 'EMA_50',
 'SMA_100',
 'EMA_100',
 'MACD',
 'bb_bbm',
 'bb_bbh',
 'bb_bbl',
 'OBV',
 'close_1',
 'open_1',
 'high_1',
 'low_1',
 'close_2',
 'open_2',
 'high_2',
 'low_2',
 'close_3',
 'open_3',
 'high_3',
 'low_3',
 'close_4',
 'open_4',
 'high_4',
 'low_4',
 'close_5',
 'open_5',
 'high_5',
 'low_5',
]
TARGET = 'Target'


def get_actual_data(
    symbol: str,
    time_frame: str = config.INTERVAL_1MINUTE,
    end_date: str = "1000 hours ago UTC",
) -> pd.DataFrame:
    client = Client()
    df = pd.DataFrame(client.get_historical_klines(symbol, time_frame, end_date))

    df = df.iloc[:, 0:6]
    df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Time", inplace=True)
    df.index = pd.to_datetime(df.index, unit="ms")
    df = df.astype(float)
    return df
