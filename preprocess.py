import argparse
from capston_polaris_v4 import *

def preprocess(df1, df2):
    
    # Correct data types
    clean_airbnb_schema(df1, inplace=True)
    clean_airbnb_schema(df2, inplace=True);

    # Concatenate the two datasets
    df_all = concat_datasets(df1, df2, df1_citycode=1, df2_citycode=0)
    
    # Drop duplicate rows
    df_all = drop_dup_rows(df_all)

    # Drop redundant columns
    df_all = drop_redunt_cols(df_all)

    # Feature engeneering
    df_all = transformation (df_all, y_col = Y_COL)

    # After feature engeneering
    #"Drop rows with null y values"

    df_all = drop_y_null(df_all, y_col = Y_COL)
    df_all.to_csv(args.out_path+"/processed.csv", index=False)
    return df_all

def df_train_test (df: pd.DataFrame, out_path: str, test_size: float, seed: int) :
    if Y_COL not in df.columns:
        raise ValueError(f"Target column '{Y_COL}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[Y_COL])
    y = df[Y_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # combine x and y again for trai and test
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(out_path+"/train.csv", index=False)
    test.to_csv(out_path+"/test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CSV")
    parser.add_argument("--csv-raw-path1", type=str, required=False, help="Path to first raw input CSV", default=DEFAULT_DATASET1_LOC)
    parser.add_argument("--csv-raw-path2", type=str, required=False, help="Path to second raw input CSV", default=DEFAULT_DATASET2_LOC)
    parser.add_argument("--out-path", type=str, default=DEFAULT_OUTPUT_LOC, help="Path to save processed CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    df1 = pd.read_csv(args.csv_raw_path1)
    df2 = pd.read_csv(args.csv_raw_path2)

    df_processed = preprocess(df1, df2)
    df_train_test(df_processed, args.out_path, args.test_size, args.seed)

    print(f"Saved processed, train and test CSVs to: {args.out_path}")
    print(f"Shape: {df_processed.shape}")






