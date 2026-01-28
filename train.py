import argparse
import json
import os
import wandb
from datetime import datetime
from capston_polaris_v4 import *
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
os.environ['WANDB_API_KEY'] = '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'

def load_data(csv_path: str) :
    train = pd.read_csv(csv_path+"/train.csv")
    test = pd.read_csv(csv_path+"/test.csv")
    X_train = train.drop(columns=[Y_COL])
    y_train = train[Y_COL]
    X_test = test.drop(columns=[Y_COL])
    y_test = test[Y_COL]
    return X_train, y_train, X_test, y_test

def build_model(seed: int, n_estimators: int) -> Pipeline:
    """
    Production-friendly baseline:
    median imputation -> RandomForestRegressor
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", rf),
        ]
    )
    return pipe


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_test: pd.DataFrame, seed: int, n_estimators: int):

    for n in range(100, n_estimators+1, 100):
        with wandb.init(project="tree-comparison", config={"n_estimators":n}):
            model = build_model(seed=seed, n_estimators=n)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            wandb.log({"n_estimators": n,
                       "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                       "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                       "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
                       "mae_test": float(mean_absolute_error(y_test, y_pred_test))})

    metrics = {
        "model": "RandomForestRegressor",
        "seed": int(seed),
        "n_estimators": int(n_estimators),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
    }

    preds_train = pd.DataFrame({"y_true": y_train, "y_pred": y_pred_train})
    preds_test = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test})

    return model, metrics, preds_train, preds_test


def save_artifacts(
    model,
    metrics: dict,
    preds_train: pd.DataFrame,
    preds_test: pd.DataFrame,
    model_out: str,
    results_dir: str,
):
    ensure_dir(os.path.dirname(model_out) or ".")
    ensure_dir(results_dir)

    joblib.dump(model, model_out)

    preds_train_path = os.path.join(results_dir, "pred_train.csv")
    preds_test_path = os.path.join(results_dir, "pred_test.csv")
    metrics_path = os.path.join(results_dir, "metrics.json")

    preds_train.to_csv(preds_train_path, index=False)
    preds_test.to_csv(preds_test_path, index=False)

    metrics_with_time = dict(metrics)
    metrics_with_time["run_timestamp"] = datetime.utcnow().isoformat() + "Z"

    with open(metrics_path, "w") as f:
        json.dump(metrics_with_time, f, indent=2)

    return preds_train_path, preds_test_path, metrics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest on processed bike dataset")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data",
        help="Path to processed CSV (output of preprocess.py)",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/model.joblib",
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save predictions and metrics",
    )

    args = parser.parse_args()

    # train the tree
    X_train, y_train, X_test, y_test = load_data(args.csv_path)
    model, metrics, preds_train, preds_test = train_and_evaluate(
        X_train, y_train, X_test, y_test, seed=args.seed, n_estimators=args.n_estimators
    )

    preds_train_path, preds_test_path, metrics_path = save_artifacts(
        model=model,
        metrics=metrics,
        preds_train=preds_train,
        preds_test=preds_test,
        model_out=args.model_out,
        results_dir=args.results_dir,
    )

    print("Saved model:", args.model_out)
    print("Saved predictions:", preds_train_path)
    print("Saved predictions:", preds_test_path)
    print("Saved metrics:", metrics_path)
    print("Metrics:", metrics)
