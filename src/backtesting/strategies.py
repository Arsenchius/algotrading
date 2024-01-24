import os
import sys
import random
import warnings
from pathlib import Path

import onnxruntime as rt
import numpy as np
import pandas as pd
import lightgbm as lgb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "models"))
from experiments import EXPERIMENT_ID_TO_FEATURES
from dataset import Data
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


warnings.filterwarnings("ignore")


# multi_model_path = '/home/kenny/algotrading/model_training/multi_model.onnx'
# binary_model_path = '/home/kenny/algotrading/model_training/binary_model.onnx'

multi_model_path = "/Users/arsenchik/Desktop/study/dipploma/machine_learning_in_hft/algotrading/models/multi_model_1_min.onnx"
multi_model_path_txt = "/Users/arsenchik/Desktop/study/dipploma/machine_learning_in_hft/algotrading/models/multi_model_1_min.txt"
binary_model_path = "/Users/arsenchik/Desktop/study/dipploma/machine_learning_in_hft/algotrading/models/binary_model_1_min.onnx"
binary_model_path_txt = "/Users/arsenchik/Desktop/study/dipploma/machine_learning_in_hft/algotrading/models/binary_model_1_min.txt"
features_24 = EXPERIMENT_ID_TO_FEATURES[24]


class TrendSwitcher(Strategy):
    n = 14

    def init(self):
        self.rsi = self.I(RSI, close, self.n)
        self.atr = self.I(ATR, close, self.n)

    def next(self):
        if crossover(self.rsi, self.atr) or crossover(self.atr, self.rsi):
            self.buy()
            # To Do
        else:
            self.sell()
            # To Do


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


class Multiple(Strategy):
    def init(self):
        self.model = lgb.Booster(model_file=multi_model_path_txt)
        self.features = features_24
        self.price_delta = 0.0005
        self.price_delta_sl = 0.0003  # 0.03%
        self.price_delta_tp = 0.0003  # 0.07%
        self.forecasts = self.I(
            lambda: np.repeat(np.nan, len(self.data)), name="forecast"
        )

    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1]

        data_for_predict = Data(self.data.df)
        data_for_predict.create_features()

        upper, lower = price * (1 + np.r_[1, -1] * self.price_delta)
        forecast_prob = self.model.predict(
            data_for_predict.df[self.features][-1:].values.astype(np.float32)
        )
        negative = forecast_prob[0][0]
        positive = forecast_prob[0][-1]

        forecast = 0
        if negative > 0.33:
            forecast = -1
        elif positive > 0.33:
            forecast = 1

        self.forecasts[-1] = forecast

        if forecast == 1 and not self.position.is_long:
            self.position.close()
            self.buy(size=0.25, tp=upper, sl=lower)
        elif forecast == -1 and not self.position.is_short:
            self.position.close()
            # self.sell(size=.25, tp=lower, sl=upper)
            self.sell(size=0.25)


class MultipleBasic(Strategy):
    def init(self):
        self.session = rt.InferenceSession(multi_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.features = features_24
        self.price_delta_sl = 0.0002  # 0.03%
        self.price_delta_tp = 0.0005  # 0.07%
        self.price_delta = 0.0007

        # Plot y for inspection
        # self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(
            lambda: np.repeat(np.nan, len(self.data)), name="forecast"
        )

    def next(self):
        # set a take profit and stop loss parameters:
        price = self.data.Close[-1]
        current_time = self.data.index[-1]
        high = self.data.High
        low = self.data.Low

        upper, lower = price * (1 + np.r_[1, -1] * self.price_delta)
        # long_upper = price * (1 + 1 * self.price_delta_tp)
        # long_lower = price * (1 - 1 * self.price_delta_sl)

        # short_upper = price * (1 + 1 * self.price_delta_sl)
        # short_lower = price * (1 - 1 * self.price_delta_tp)

        # data preparation:
        data_for_predict = Data(self.data.df)
        data_for_predict.create_features()

        # print(data_for_predict.df)

        # create a model prediction:
        # Forecast the next movement
        forecast = self.session.run(
            None,
            {
                self.input_name: data_for_predict.df[self.features][-1:].values.astype(
                    np.float32
                )
            },
        )[0][0]
        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast

        # define a tips based on model predictions:
        # if forecast == 1:
        #     if self.position.is_short or not self.position:
        #         # self.position.close()
        #         self.buy(size=.25, tp=long_upper, sl=long_lower)
        #         # self.buy()
        # elif forecast == -1:
        #     if self.position.is_long or not self.position:
        #         # self.position.close()
        #         self.sell(size=.25, tp=short_lower, sl=short_upper)
        #         # self.sell()

        if forecast == 1 and not self.position.is_long:
            self.buy(size=0.25, tp=upper, sl=lower)
        elif forecast == -1 and not self.position.is_short:
            self.sell(size=0.25, tp=lower, sl=upper)

        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta("3 minutes"):
                # if trade.is_long:
                #     trade.sl = max(trade.sl, low)
                # else:
                #     trade.sl = min(trade.sl, high)
                self.position.close()


class Binary(Strategy):
    def init(self):
        self.price_delta = 0.0006  # 0.06%
        self.forecasts = self.I(
            lambda: np.repeat(np.nan, len(self.data)), name="forecast"
        )

    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1]
        upper, lower = price * (1 + np.r_[1, -1] * self.price_delta)
        positive = self.data.predicted_proba[-1]

        forecast = -1
        if positive > 0.55:
            forecast = 1
        if 1 - positive > 0.55:
            forecast = 0

        self.forecasts[-1] = forecast

        if forecast == 1 and not self.position.is_long:
            self.position.close()
            self.buy(size=0.25, tp=upper, sl=lower)
        elif forecast == 0 and not self.position.is_short:
            self.position.close()
            self.sell(size=0.25)

        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta(minutes=45):
                self.position.close()


class BinaryBasic(Strategy):
    price_delta = 0.004

    def init(self):
        self.session = rt.InferenceSession(binary_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.features = features_24
        # self.price_delta = price_delta # 0.05%

        # Plot y for inspection
        # self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(
            lambda: np.repeat(np.nan, len(self.data)), name="forecast"
        )

    def next(self):
        # set a take profit and stop loss parameters:
        price = self.data.Close[-1]
        upper, lower = price * (1 + np.r_[1, -1] * self.price_delta)
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        # data preparation:
        data_for_predict = Data(self.data.df)
        data_for_predict.create_features()

        # create a model prediction:
        # Forecast the next movement
        forecast_prob = self.session.run(
            [self.output_name],
            {
                self.input_name: data_for_predict.df[self.features][-1:].values.astype(
                    np.float32
                )
            },
        )[0][-1]
        forecast = forecast_prob
        self.forecasts[-1] = forecast

        # define a tips based on model predictions:
        if forecast == 1:
            if self.position.is_short or not self.position:
                self.position.close()
                # self.buy(size=.1,tp=upper, sl=lower)
                self.buy()
        elif forecast == 0:
            if self.position.is_long or not self.position:
                self.position.close()
                # self.sell(size=.1,tp=lower, sl=upper)
                self.sell()

        # for trade in self.trades:
        #     if current_time - trade.entry_time > pd.Timedelta('2 minutes'):
        #         if trade.is_long:
        #             trade.sl = max(trade.sl, low)
        #         else:
        #             trade.sl = min(trade.sl, high)


class Random(Strategy):
    def init(self):
        self.forecasts = self.I(
            lambda: np.repeat(np.nan, len(self.data)), name="forecast"
        )

    def next(self):
        forecast = random.choice([0, 1])
        self.forecasts[-1] = forecast

        if forecast == 1 or not self.position:
            self.position.close()
            self.buy(size=0.25)
        elif forecast == -1 or not self.position:
            self.position.close()
            self.sell(size=0.25)
