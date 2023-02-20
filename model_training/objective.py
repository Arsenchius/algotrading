from model import *
from data import *
from sklearn.model_selection import TimeSeriesSplit


def Objective(model, score_metric, number_of_splits=5, test_size=5*24*60):
        """
        Define an objective function to be minimized or maximized.
        type:
        - int: integer
        - uni: a uniform float sampling
        - log: a uniform float sampling on log scale
        - dis: a discretized uniform float sampling
        - cat: category; ('auto', 'mode1', 'mode2', )
        """
        method_names = {
            "int": "suggest_int",
            "uni": "suggest_uniform",
            "log": "suggest_loguniform",
            "dis": "suggest_discrete_uniform",
            "cat": "suggest_categorical",
            "float": "suggest_float",
        }
        model_params = {
            model.model_name: {
                key: (
                    method_names.get(val[0]),
                    ("{}".format(key), *val[1:]),
                )
                if type(val) is tuple else val
                for key, val in model.params.items()
            }
        }


        def _objective(trial):

            # Invoke suggest methods of a Trial object to generate hyperparameters.
            params = {}
            for key, val in model_params[model.model_name].items():
                if type(val) is tuple:
                    if val[0].split('_')[1] == "float" and len(val[1]) > 3:
                        values = list(val[1])
                        params[key] = getattr(trial, val[0])(*tuple(values[:-1]), step=values[-1])
                    else:
                        params[key] = getattr(trial, val[0])(*val[1])
                else:
                    params[key] = val

            #  evaluation
            tss = TimeSeriesSplit(n_splits=number_of_splits, test_size=test_size)
            model.df = model.df.sort_index()
            predictions = []
            scores = []

            for train_idx, val_idx in tss.split(model.df):
                train = model.df.iloc[train_idx]
                test = model.df.iloc[val_idx]

                X_train = train[FEATURES]
                y_train = train[TARGET]

                X_test = test[FEATURES]
                y_test = test[TARGET]

                model_loaded = model.model(**params)

                if model.model_name in ["LightGBM", "XGBoost"]:
                    model_loaded.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_test, y_test)],
                    )
                else:
                    model_loaded.fit(X_train, y_train)

                y_pred = model_loaded.predict(X_test)
                predictions.append(y_pred)
                scores.append(score_metric(y_test, y_pred))

            return np.mean(scores)

        return _objective
