from typing import NoReturn
import pandas as pd
import ta
import numpy as np


class Data:
    def __init__(self, df: pd.DataFrame) -> NoReturn:
        self.df = df

    def create_features_and_target(
        self, task_type: str, period: int = 1, indicators: bool = True
    ) -> NoReturn:
        if indicators:
            self.indicators_calc()
        self.df = self.df[:-period]
        for i in range(5):
            self.df[f"close_{i+1}"] = self.df["Close"].shift(i + 1)
            self.df[f"open_{i+1}"] = self.df["Open"].shift(i + 1)
            self.df[f"high_{i+1}"] = self.df["High"].shift(i + 1)
            self.df[f"low_{i+1}"] = self.df["Low"].shift(i + 1)
        self.df["Target"] = self.df["Close"].shift(-period) - self.df["Close"]
        if task_type == "classification":
            self.df["Target"] = np.where(self.df["Target"] > 0, 1, 0)
        self.df.dropna(inplace=True)

    def indicators_calc(self) -> NoReturn:
        periods = [3, 5, 15, 30, 50, 100]

        # RSI calculating for window = 14
        self.df[f"RSI_{14}"] = ta.momentum.rsi(self.df["Close"], window=14)

        # SMA and EMA calc for period in periods
        for period in periods:
            self.df[f"SMA_{period}"] = self.df["Close"].rolling(period).mean()
            self.df[f"EMA_{period}"] = ta.trend.ema_indicator(
                self.df["Close"], window=period
            )

        # MACD calc with signal EMA_9, slow EMA_26 and fast EMA_12
        self.df["MACD"] = ta.trend.macd_signal(
            self.df["Close"], window_slow=26, window_fast=12, window_sign=9
        )

        indicator_bb = ta.volatility.BollingerBands(
            close=self.df["Close"], window=20, window_dev=2
        )

        # Add Bollinger Bands features
        self.df["bb_bbm"] = indicator_bb.bollinger_mavg()
        self.df["bb_bbh"] = indicator_bb.bollinger_hband()
        self.df["bb_bbl"] = indicator_bb.bollinger_lband()

        # Add Bollinger Band high indicator
        # self.df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

        # Add Bollinger Band low indicator
        # self.df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

        # Add on-balance volume indicator
        self.df["OBV"] = ta.volume.on_balance_volume(self.df['Close'], self.df['Volume'])


        # self.df.dropna(inplace=True)
