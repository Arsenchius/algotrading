from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    accuracy_score,
    r2_score,
    mean_absolute_error,
)

# 'n_estimators': ('int', 1, 100),
#         'max_depth': ('dis', 1, 100, 5),
#         'random_state': 128

param_grid = {
    "LightGBM": {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": ("int", 50, 1000),
        "learning_rate": ("float", 0.01, 0.3),
        "num_leaves": ("int", 20, 3000, 20),
        "max_depth": ("int", 3, 12),
        "min_data_in_leaf": ("int", 200, 10000, 100),
        "lambda_l1": ("int", 0, 100, 5),
        "lambda_l2": ("int", 0, 100, 5),
        "min_gain_to_split": ("float", 0, 15),
        "bagging_fraction": ("float", 0.2, 0.95, 0.1),
        "bagging_freq": ("cat", [1]),
        "feature_fraction": ("float", 0.2, 0.95, 0.1),
    },
    "XGBoost": {
        "max_depth": ("int", 3, 12),
        "learning_rate": ("float", 0.01, 0.3),
        "n_estimators": ("int", 50, 500),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("log", 1e-8, 1.0),
        "subsample": ("log", 0.01, 1.0),
        "colsample_bytree": ("log", 0.01, 1.0),
        "reg_alpha": ("log", 1e-8, 1.0),
        "reg_lambda": ("log", 1e-8, 1.0),
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    },
    # "CatBoost": {
    #     "loss_function": "RMSE",
    #     "task_type": "GPU",
    #     "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10.0),
    #     "max_bin": trial.suggest_int("max_bin", 200, 400),
    #     #'rsm': trial.suggest_uniform('rsm', 0.3, 1.0),
    #     "subsample": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
    #     "learning_rate": trial.suggest_uniform("learning_rate", 0.006, 0.018),
    #     "n_estimators": 25000,
    #     "max_depth": trial.suggest_int("max_depth", 3, 12),
    #     "random_state": trial.suggest_categorical("random_state", [2020]),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
    # },
}

metric_grid = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r_2": r2_score,
    "log_loss": log_loss,
    "accuracy": accuracy_score,
    "roc_auc": roc_auc_score,
    "f1": f1_score,
}
