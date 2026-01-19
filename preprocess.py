import argparse
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps.
    Keep this function pure: df in -> df out.
    """
    df = df.copy()

    # 1) Standardize column names (safe habit)
    df.columns = [c.strip() for c in df.columns]

    # 2) Drop exact duplicate rows
    df = df.drop_duplicates()

    # 3) Parse date and create features (bike dataset has 'dteday')
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

        # If parsing failed for some rows, drop them (simple + explicit)
        df = df.dropna(subset=["dteday"])

        # Feature engineering from date (keeps model-friendly columns)
        df["dteday_year"] = df["dteday"].dt.year
        df["dteday_month"] = df["dteday"].dt.month
        df["dteday_dayofweek"] = df["dteday"].dt.dayofweek

        # Drop the raw date column (presentation-friendly, avoids non-numeric raw text)
        df = df.drop(columns=["dteday"])

    # 4) Handle missing values (simple strategy, presentation-aligned)
    # For numeric columns: fill with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # For non-numeric columns: fill with "unknown"
    obj_cols = df.select_dtypes(exclude=["number"]).columns
    for c in obj_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("unknown")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CSV")
    parser.add_argument("--csv-raw-path", type=str, required=True, help="Path to raw input CSV")
    parser.add_argument("--out-path", type=str, default="data/processed.csv", help="Path to save processed CSV")

    args = parser.parse_args()

    df = pd.read_csv(args.csv_raw_path)
    df_processed = preprocess(df)
    df_processed.to_csv(args.out_path, index=False)

    print(f"Saved processed CSV to: {args.out_path}")
    print(f"Shape: {df_processed.shape}")
