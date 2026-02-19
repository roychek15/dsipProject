import argparse
import os
import joblib
import pandas as pd
from capston_polaris_v4 import *

TARGET_COL = "review_scores_rating"


def load_model(model_path: str):
    return joblib.load(model_path)


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict with a trained sklearn pipeline/model.

    - If TARGET_COL exists, we keep it in the output for convenience,
      but we do NOT use it as a feature.
    """
    df = df.copy()

    y_pred = model.predict(df)

    out = pd.DataFrame({"y_pred": y_pred})

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a saved model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/random_forest.joblib",
        help="Path to saved model (output of train.py)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/test.csv",
        help="Path to input test CSV",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="results/predictions.csv",
        help="Where to save predictions CSV",
    )

    args = parser.parse_args()

    model = load_model(args.model_path)
    df = load_data(args.csv_path)

    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
    y = df[TARGET_COL] if TARGET_COL in df.columns else None

    X = X.drop(columns=[SAMPLE_WEIGHT_COL]) # we save the sample weights column along, but we do not want to predict based on it.

    preds = predict(model, X)

    ensure_dir(os.path.dirname(args.out_path) or ".")
    preds.to_csv(args.out_path, index=False)

    print("Saved predictions:", args.out_path)
    print("Shape:", preds.shape)

    
    if y is not None:
       
    # Can calculate metrics only in cases where the original Y value was not missing
      y_na_idx = y.isna()

      metric = compute_metrics(y_true = y[~y_na_idx], y_pred=preds[~y_na_idx])
    
      print("Model metrics:")
      print(metric)
    
