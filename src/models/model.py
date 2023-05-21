import os
import sys
import yaml
import warnings
from pathlib import Path
from typing import NoReturn, List, Dict, Union

import ta
import optuna
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from optuna.integration import LightGBMPruningCallback
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "data"))
sys.path.append(os.path.join(Path(os.path.dirname(SCRIPT_DIR)), "configs"))
from data import get_actual_data, FEATURES, TARGET
from dataset import Data
from metrics import metric_grid
from objective import Objective


warnings.filterwarnings("ignore")


def get_metrics() -> Dict:
    metrics_config_file = "/Users/arsenchik/Desktop/dipploma/machine_learning_in_hft/algotrading/src/configs/metrics.yaml"
    try:
        with open(metrics_config_file, "r") as file:
            metrics = yaml.safe_load(file)
    except Exception as e:
        print("Error reading the config file")
    return metrics


def get_params(model_name: str) -> Dict:
    model_config = "/Users/arsenchik/Desktop/dipploma/machine_learning_in_hft/algotrading/src/configs/model_config.yaml"
    try:
        with open(model_config) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print("Error reading the config file")

    return params["model_name"][model_name]["parameters"]


class Model:
    def __init__(self, model_name) -> NoReturn:
        self.model_name = model_name
        self.params = get_params(model_name)
        self.model_trained = None

        if model_name == "LightGBM":
            self.model_classification = lgb.LGBMClassifier
            self.model_regression = lgb.LGBMRegressor
        elif model_name == "XGBoost":
            self.model_classification = xgb.XGBClassifier
            self.model_regression = xgb.XGBRegressor
        elif model_name == "RandomForest":
            self.model_classification = RandomForestClassifier
            self.model_regression = RandomForestRegressor
        else:
            print("wrong type of model")

    def train(
        self,
        data: Data,
        metric: str,
        features: List[str],
        type_of_training: str = "default",
        is_optimized: bool = False,
    ):
        """
        type_of_training:
        default - if default timeseries train test split
        alternative - if another one, when we dont use too old part of training data
        """

        def define_task_type() -> str:
            if target_values == 2 or target_values == 3:
                return "classification"
            else:
                return "regression"

        target_values = len(set(data.df["Target"].values))
        task_type = define_task_type()
        metrics = get_metrics()

        if type_of_training == "alternative":
            return None
        else:
            params = {}
            if is_optimized:
                curr_cwd = os.getcwd()
                optimization_results = os.path.join(curr_cwd, "optimization_results")
                optimization_dir_path = os.path.join(
                    optimization_results, f"res_{task_type}"
                )
                cur_model_params = os.path.join(
                    optimization_dir_path,
                    f"{self.model_name}_best_parameters_{metric}.yaml",
                )
                with open(cur_model_params, "r") as file:
                    load_params = yaml.safe_load(file)
                del load_params["best_value"]
                del load_params["metric"]
                params = load_params

            if task_type == "classification":
                model_loaded = self.model_classification(**params)
            elif task_type == "regression":
                model_loaded = self.model_regression(**params)
            else:
                print("Error with task type")
                return None
            X_all = data.df[features]
            y_all = data.df["Target"]
            model_loaded.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=-1)

            if target_values == 3:
                if metric == "roc_auc":
                    y_pred = model_loaded.predict_proba(X_all)
                    score = metric_grid[metric](y_all, y_pred, multi_class="ovr")
                else:
                    y_pred = model_loaded.predict(X_all)
                    if metric == "f1":
                        score = metric_grid[metric](y_all, y_pred, average="macro")
                    else:
                        score = metric_grid[metric](y_all, y_pred)
            else:
                y_pred = model_loaded.predict(X_all)
                score = metric_grid[metric](y_all, y_pred)
            self.model_trained = model_loaded
        return score

    def optimize(
        self,
        data: Data,
        output_dir_path: Union[str, Path],
        features: List[str],
        metric: str = None,
        number_of_trials: int = 10,
    ) -> NoReturn:
        is_metric = False
        metrics = get_metrics()
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        is_binary = False

        def define_task_type() -> str:
            if len(set(data.df["Target"].values)) <= 10:
                if len(set(data.df["Target"].values)) == 2:
                    is_binary = True
                return "classification"
            else:
                return "regression"

        task_type = define_task_type()

        full_dir_output_path = os.path.join(output_dir_path, f"res_{task_type}/")

        if not os.path.exists(full_dir_output_path):
            os.mkdir(full_dir_output_path)

        if metric is not None:
            is_metric = True

        for item in tqdm(metrics[task_type]):
            if is_metric:
                if item != metric:
                    continue
            if item != "roc_auc":
                continue
            direction = metrics[item]["objective"]
            direction = "minimize" if direction == "min" else "maximize"
            study_name = f"{task_type} {self.model_name} with metric - {item}"
            study = optuna.create_study(direction=direction, study_name=study_name)
            func = Objective(self, data, item, features, task_type, is_binary=is_binary)
            study.optimize(func, n_trials=number_of_trials)
            results = study.best_params
            results["best_value"] = study.best_value
            results["metric"] = metrics[item]["abbr"]
            full_path = os.path.join(
                full_dir_output_path, f"{self.model_name}_best_parameters_{item}.yaml"
            )
            with open(full_path, "w") as outfile:
                yaml.dump(results, outfile, default_flow_style=False)
