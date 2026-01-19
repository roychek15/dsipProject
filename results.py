import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_preds(pred_path: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pred_path} must contain columns {required}, got {set(df.columns)}")
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}


def save_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("residual (y_true - y_pred)")
    plt.ylabel("count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions and save simple plots")
    parser.add_argument(
        "--pred-path",
        type=str,
        default="results/pred_test.csv",
        help="Path to predictions CSV with columns y_true,y_pred",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Directory to save report and plots",
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = load_preds(args.pred_path)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    metrics = compute_metrics(y_true, y_pred)

    # Save metrics
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    scatter_path = os.path.join(args.out_dir, "pred_vs_true.png")
    resid_path = os.path.join(args.out_dir, "residuals_hist.png")

    save_scatter(y_true, y_pred, scatter_path)
    save_residual_hist(y_true, y_pred, resid_path)

    print("Results metrics:", metrics)
    print("Saved report:", report_path)
    print("Saved plot:", scatter_path)
    print("Saved plot:", resid_path)
