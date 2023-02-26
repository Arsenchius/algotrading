from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import onnxruntime as rt
import os
import sys
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "model_training"))
from experiments import EXPERIMENT_ID_TO_FEATURES
from dataset import Data
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


model_path = '/home/kenny/algotrading/model_training/model.onnx'
binary_model_path = '/home/kenny/algotrading/model_training/binary_model.onnx'
features_24 = EXPERIMENT_ID_TO_FEATURES[24]

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


class Basic(Strategy):

    def init(self):
        self.session = rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.features = features_24
        self.price_delta = .005 # 0.05%

        # Plot y for inspection
        # self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

    def next(self):

        # set a take profit and stop loss parameters:
        price = self.data.Close[-1]
        upper, lower = price * (1 + np.r_[1, -1]*self.price_delta)

        # data preparation:
        # print(self.data.df[-1:])
        # print(self.data.df.drop(['Close_Time'], axis=1))
        data_for_predict = Data(self.data.df)
        data_for_predict.create_features()

        # print(data_for_predict.df)

        # create a model prediction:
        # Forecast the next movement
        forecast = self.session.run( None, {self.input_name: data_for_predict.df[self.features][-1:].values.astype(np.float32)})[0][0]

        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast

        # define a tips based on model predictions:
        if forecast == 1:
            if self.position.is_short or not self.position:
                self.position.close()
                self.buy(tp=upper, sl=lower)
                # self.buy()
        elif forecast == -1:
            if self.position.is_long or not self.position:
                self.position.close()
                self.sell(tp=lower, sl=upper)
                # self.sell()

        # if forecast == 1 and not self.position.is_long:
        #     # self.buy(size=.2, tp=upper, sl=lower)
        #     self.buy()
        # elif forecast == -1 and not self.position.is_short:
        #     # self.sell(size=.2, tp=lower, sl=upper)
        #     self.sell()



class BinaryBasic(Strategy):

    price_delta = .004

    def init(self):
        self.session = rt.InferenceSession(binary_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[1].name
        self.features = features_24
        # self.price_delta = price_delta # 0.05%

        # Plot y for inspection
        # self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')


    def next(self):

        # set a take profit and stop loss parameters:
        price = self.data.Close[-1]
        upper, lower = price * (1 + np.r_[1, -1]*self.price_delta)
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        # data preparation:
        data_for_predict = Data(self.data.df)
        data_for_predict.create_features()

        # print(data_for_predict.df)

        # create a model prediction:
        # Forecast the next movement
        forecast_prob = self.session.run( [self.output_name], {self.input_name: data_for_predict.df[self.features][-1:].values.astype(np.float32)})[0][-1]
        if forecast_prob[1] >= 0.45:
            forecast = 1
        else:
            forecast = 0
        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast

        # define a tips based on model predictions:
        if forecast == 1:
            if self.position.is_short or not self.position:
                self.position.close()
                self.buy(size=.1,tp=upper, sl=lower)
                # self.buy()
        elif forecast == 0:
            if self.position.is_long or not self.position:
                self.position.close()
                self.sell(size=.1,tp=lower, sl=upper)
                # self.sell()

        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 minutes'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)
                else:
                    trade.sl = min(trade.sl, high)
