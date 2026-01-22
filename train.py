import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


TARGET_COL = "review_scores_rating"


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


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


def train_and_evaluate(df: pd.DataFrame, seed: int, test_size: float, n_estimators: int):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = build_model(seed=seed, n_estimators=n_estimators)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model": "RandomForestRegressor",
        "seed": int(seed),
        "test_size": float(test_size),
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
        default="data/bike_day_processed.csv",
        help="Path to processed CSV (output of preprocess.py)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)

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

    df = load_data(args.csv_path)
    model, metrics, preds_train, preds_test = train_and_evaluate(
        df=df, seed=args.seed, test_size=args.test_size, n_estimators=args.n_estimators
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
