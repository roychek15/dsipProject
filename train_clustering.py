import wandb
import joblib
from capston_polaris_v4 import *
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


class KmeansCluster:
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_val: pd.DataFrame, y_val: pd.DataFrame, seed: int):

        best_mse = float("inf")



        X_train=X_train.drop(SAMPLE_WEIGHT_COL, axis=1).copy()
        X_val=X_val.drop(SAMPLE_WEIGHT_COL, axis=1).copy()

        imputer = SimpleImputer(strategy="median")  # or "mean"
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        for k in range(10, 21):
            with wandb.init(project="kmeans-comparison", config={"k": k}):
                km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                train_clusters = km.fit_predict(X_train_imp)

                # cluster -> mean(y_train)
                cluster_mean = np.array([y_train[train_clusters == j].mean() for j in range(k)])

                # predict on validation
                val_clusters = km.predict(X_val_imp)
                y_val_pred = np.array([cluster_mean[c] for c in val_clusters])

                mse = mean_squared_error(y_val, y_val_pred)
                if mse < best_mse:
                    best_mse = mse
                    self._best_model = km
                    best_y_predict = y_val_pred

                # Log everything to compare in the dashboard
                wandb.log({
                    "n_clusters": k,
                    "val_rmse": mse,
                    "inertia": km.inertia_
                })

        self._metrics = {
            "model": "Kmeans",
            "seed": int(seed),
            "n_clusters": int(self._best_model.n_clusters),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "val_rmse": float(np.sqrt(best_mse)),
            "val_r2": float(r2_score(y_val, best_y_predict)),
        }
        return self._best_model, self._metrics

    def save_model(self, model_dir: str, results_dir: str):
        model_path = os.path.join(model_dir, "kmeans.joblib")
        joblib.dump(self._best_model, model_path, compress=("lz4", 3))

        metrics_path = os.path.join(results_dir, "kmeans.json")
        metrics_with_time = dict(self._metrics)
        metrics_with_time["run_timestamp"] = datetime.utcnow().isoformat() + "Z"

        with open(metrics_path, "w") as f:
            json.dump(metrics_with_time, f, indent=2)
