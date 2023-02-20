from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    accuracy_score,
    r2_score,
    mean_absolute_error,
)

metric_grid = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r_2": r2_score,
    "log_loss": log_loss,
    "accuracy": accuracy_score,
    "roc_auc": roc_auc_score,
    "f1": f1_score,
}
