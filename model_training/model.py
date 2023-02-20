import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np

# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    accuracy_score,
    r2_score,
    mean_absolute_error,
)
from data import *
from optuna.integration import LightGBMPruningCallback
import optuna
from metrics import *
from objective import Objective
import yaml
# from tqdm import tqdm
from tqdm.notebook import tqdm
import ta
import os



def get_metrics():
    metrics_config_file = "metrics.yaml"
    try:
        with open(metrics_config_file, "r") as file:
            metrics = yaml.safe_load(file)
    except Exception as e:
        print("Error reading the config file")
    return metrics


def get_params(model_name):
    try:
        with open('config.yaml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print("Error reading the config file")

    return params['model_name'][model_name]['parameters']


class Model:
    def __init__(self, df, model_name, task_type):
        self.df = df
        self.model_name = model_name
        self.task_type = task_type
        if self.task_type == "classification":
            if model_name == "LightGBM":
                self.model = lgb.LGBMClassifier
            elif model_name == "XGBoost":
                self.model = xgb.XGBClassifier
            # elif model_name == "Catboost":
            # self.model = CatBoostClassifier()
            elif model_name == "RandomForest":
                self.model = RandomForestClassifier
            # elif model_name == "Linear":
            #     self.model =
            else:
                print("wrong type of model")
        elif self.task_type == "regression":
            if model_name == "LightGBM":
                self.model = lgb.LGBMRegressor
            elif model_name == "XGBoost":
                self.model = xgb.XGBRegressor
            # elif model_name == "Catboost":
            # self.model = CatBoostRegressor()
            elif model_name == "RandomForest":
                self.model = RandomForestRegressor
            else:
                print("wrong type of model")
        else:
            print("wrong type of task")

        self.params = get_params(model_name)


    def indicators_calc(self):
        for i in range(2, 10, 2):
            self.df[f"RSI_{i}"] = ta.momentum.rsi(self.df["Close"], window=i)
            self.df[f"SMA_{i*10}"] = self.df["Close"].rolling(i * 10).mean()
        # df['MACD'] = ta.trend.macd_diff(df['Close'])
        self.df.dropna(inplace=True)


    def create_target(self, period=1):
        self.df["Target"] = self.df["Close"].shift(-period) - self.df["Close"]
        self.df = self.df[:-period]
        if self.task_type == "classification":
            self.df["Target"] = np.where(self.df["Target"] > 0, 1, 0)


    def optimize(self, output_dir_path, metric=None, number_of_trials=10):
        is_metric = False
        metrics = get_metrics()
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        full_dir_output_path = os.path.join(output_dir_path, f"res_{self.task_type}/")

        if not os.path.exists(full_dir_output_path):
            os.mkdir(full_dir_output_path)

        if metric is not None:
            is_metric = True

        for item in tqdm(metrics[self.task_type]):
            if is_metric:
                if item != metric:
                    continue
            direction = metrics[item]["objective"]
            direction = "minimize" if direction == "min" else "maximize"
            study_name = f"{self.task_type} {self.model_name} with metric - {item}"
            study = optuna.create_study(direction=direction, study_name=study_name)
            # func = lambda trial: Objective(self, metric_grid[item])
            func = Objective(self, metric_grid[item])
            study.optimize(func, n_trials=number_of_trials)
            results = study.best_params
            results["best_value"] = study.best_value
            results["metric"] = metrics[item]['abbr']
            full_path = os.path.join(
                full_dir_output_path, f"{self.model_name}_best_parameters_{item}.yaml"
            )
            with open(full_path, "w") as outfile:
                yaml.dump(results, outfile, default_flow_style=False)
