import wandb
from capston_polaris_v4 import *
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score


class BasicModel:
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_val: pd.DataFrame, y_val: pd.DataFrame, seed: int):

        best_mse = float("inf")

        baseline = y_train.mean()
        y_val_pred = np.full(len(y_val), baseline)
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        with wandb.init(project="basic-model"):
                # Log everything to compare in the dashboard
                wandb.log({
                    "y_train_mean": baseline,
                    "val_rmse": np.sqrt(mse),
                    "val_r2": r2
                })

        self._metrics = {
            "model": "BasicModel(y-mean)",
            "seed": int(seed),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "y_train_mean": int(baseline),
            "val_rmse": float(np.sqrt(mse)),
            "val_r2": float(r2),
        }
        return self._metrics

    def save_model(self, model_dir: str, results_dir: str):
        metrics_path = os.path.join(results_dir, "BasicModel.json")
        metrics_with_time = dict(self._metrics)
        metrics_with_time["run_timestamp"] = datetime.utcnow().isoformat() + "Z"

        with open(metrics_path, "w") as f:
            json.dump(metrics_with_time, f, indent=2)
