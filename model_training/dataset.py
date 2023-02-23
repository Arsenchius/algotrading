from typing import NoReturn
import pandas as pd
import ta
import numpy as np

class Data:
    def __init__(self, df: pd.DataFrame) -> NoReturn:
        self.df = df

    def indicators_calc(self) -> NoReturn:
        for i in range(2, 10, 2):
            self.df[f"RSI_{i}"] = ta.momentum.rsi(self.df["Close"], window=i)
            self.df[f"SMA_{i*10}"] = self.df["Close"].rolling(i * 10).mean()
        # df['MACD'] = ta.trend.macd_diff(df['Close'])
        self.df.dropna(inplace=True)

    def create_target(self, task_type: str, period: int = 1) -> NoReturn:
        self.df["Target"] = self.df["Close"].shift(-period) - self.df["Close"]
        self.df = self.df[:-period]
        if task_type == "classification":
            self.df["Target"] = np.where(self.df["Target"] > 0, 1, 0)
