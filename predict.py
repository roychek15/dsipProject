import argparse
import os
import joblib
import pandas as pd


TARGET_COL = "cnt"


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

    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
    y_pred = model.predict(X)

    out = pd.DataFrame({"y_pred": y_pred})

    # Optional: keep y_true alongside predictions if available
    if TARGET_COL in df.columns:
        out.insert(0, "y_true", df[TARGET_COL].values)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a saved model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to saved model (output of train.py)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to input CSV (typically processed CSV)",
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

    preds = predict(model, df)

    ensure_dir(os.path.dirname(args.out_path) or ".")
    preds.to_csv(args.out_path, index=False)

    print("Saved predictions:", args.out_path)
    print("Shape:", preds.shape)
