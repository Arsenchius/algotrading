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
from tqdm import tqdm
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

    # def objective(
    #     self, trial, param_grid, score_metric, number_of_splits=5, test_size=5 * 24 * 60
    # ):

    #     tss = TimeSeriesSplit(n_splits=number_of_splits, test_size=test_size)
    #     self.df = self.df.sort_index()
    #     predictions = []
    #     scores = []

    #     for train_idx, val_idx in tss.split(self.df):
    #         train = self.df.iloc[train_idx]
    #         test = self.df.iloc[val_idx]

    #         X_train = train[FEATURES]
    #         y_train = train[TARGET]

    #         X_test = test[FEATURES]
    #         y_test = test[TARGET]

    #         self.model.fit(
    #             X_train,
    #             y_train,
    #             eval_set=[(X_test, y_test)],
    #             early_stopping_rounds=100,
    #         )

    #         y_pred = self.model.predict(X_test)
    #         predictions.append(y_pred)
    #         scores.append(score_metric(y_test, y_pred))

    #     return np.mean(scores)

    def optimize(self, output_dir_path):
        metrics = get_metrics()
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        full_dir_output_path = os.path.join(output_dir_path, f"res_{self.task_type}/")

        if not os.path.exists(full_dir_output_path):
            os.mkdir(full_dir_output_path)

        for item in tqdm(metrics[self.task_type]):
            direction = metrics[item]["objective"]
            direction = "minimize" if direction == "min" else "maximize"
            study_name = f"{self.task_type} {self.model_name} with metric - {item}"
            study = optuna.create_study(direction=direction, study_name=study_name)
            # func = lambda trial: Objective(self, metric_grid[item])
            func = Objective(self, metric_grid[item])
            study.optimize(func, n_trials=10)
            results = study.best_params
            results["best_value"] = study.best_value
            results["metric"] = metrics[item]['abbr']
            full_path = os.path.join(
                full_dir_output_path, f"{self.model_name}_best_parameters_{item}.yaml"
            )
            with open(full_path, "w") as outfile:
                yaml.dump(results, outfile, default_flow_style=False)
