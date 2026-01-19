import argparse
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps.
    Keep this function pure: df in -> df out.
    """
    # Example steps (replace with your notebook logic later)
    df = df.drop_duplicates()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CSV")
    parser.add_argument(
        "--csv-raw-path",
        type=str,
        required=True,
        help="Path to raw input CSV"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/processed.csv",
        help="Path to save processed CSV"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_raw_path)
    df_processed = preprocess(df)
    df_processed.to_csv(args.out_path, index=False)
