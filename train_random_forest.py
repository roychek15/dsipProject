import wandb
import joblib
from datetime import datetime
from capston_polaris_v4 import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

class RandomForest:
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                           y_val: pd.DataFrame, seed: int):

        rf = RandomForestRegressor(random_state=seed, n_jobs=-1)

        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [10, 12],
            "max_features": [10],
            "min_samples_leaf": [5, 7],
            "max_samples": [0.85]
        }

        gridSearch = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="neg_mean_squared_error",
                                    cv=5, n_jobs=-1, verbose=1)
        
        sample_weight = X_train[SAMPLE_WEIGHT_COL]
        X_train=X_train.drop(SAMPLE_WEIGHT_COL, axis=1).copy()
        X_val=X_val.drop(SAMPLE_WEIGHT_COL, axis=1).copy()
        gridSearch.fit(X_train, y_train, sample_weight=sample_weight)
        self._best_model = gridSearch.best_estimator_
        p = gridSearch.best_params_

        with wandb.init(project="tree-comparison"):
            y_pred_train = self._best_model.predict(X_train)
            y_pred_val = self._best_model.predict(X_val)
            wandb.log({"n_estimators": p["n_estimators"],
                       "max_depth": p["max_depth"],
                       "min_samples_leaf":p["min_samples_leaf"],
                       "max_features":p["max_features"],
                       "max_samples":p["max_samples"],
                       "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                       "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                       "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                       "val_mae": float(mean_absolute_error(y_val, y_pred_val))})

        # keep metrics of best model
        self._metrics = {
            "model": "RandomForestRegressor",
            "seed": int(seed),
            "n_estimators": int(p["n_estimators"]),
            "max_depth": int(p["max_depth"]),
            "min_samples_leaf":p["min_samples_leaf"],
            "max_features":p["max_features"],
            "max_samples":p["max_samples"],
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "val_r2": float(r2_score(y_val, y_pred_val)),
        }

        return self._best_model, self._metrics

    def save_model(self, model_dir: str, results_dir: str):
        model_path = os.path.join(model_dir, "random_forest.joblib")
        joblib.dump(self._best_model, model_path, compress=("lz4", 3))

        metrics_path = os.path.join(results_dir, "random_forest_metrics.json")
        metrics_with_time = dict(self._metrics)
        metrics_with_time["run_timestamp"] = datetime.utcnow().isoformat() + "Z"

        with open(metrics_path, "w") as f:
            json.dump(metrics_with_time, f, indent=2)
