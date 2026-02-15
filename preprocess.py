import argparse

from capston_polaris_v4 import *

TARGET_COL = "review_scores_rating"


def df_train_test (df: pd.DataFrame, out_path: str, test_size: float, seed: int) :
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

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
    parser.add_argument("--csv-raw-path1", type=str, required=True, help='Type "default" for default dataset')
    parser.add_argument("--csv-raw-path2", type=str, required=False, help='Path to second raw input CSV, type "default" for default dataset')
    parser.add_argument("--csv-raw-path3", type=str, required=False, help='Path to third raw input CSV,type "default" for default dataset')
    parser.add_argument("--out-path", type=str, default=DEFAULT_OUTPUT_LOC, help="Path to save processed CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--use-genAI", required=False,type=lambda x: x.lower() == "true")

    args = parser.parse_args()

    if args.csv_raw_path1 == "default":
      df = pd.read_csv(DEFAULT_DATASET1_LOC)
    else:
      df = pd.read_csv(args.csv_raw_path1)

    df = drop_y_null(df, TARGET_COL )     # Drop rows with null y values
    df_processed = preprocess(df, args.use_genAI)

    if args.csv_raw_path2 == "default":
      df2 = pd.read_csv(DEFAULT_DATASET2_LOC)
      df2 = drop_y_null(df2, TARGET_COL )     # Drop rows with null y values
      df2 = preprocess(df2,  args.use_genAI)
      df_processed = pd.concat([df_processed, df2], axis=0)
    elif args.csv_raw_path2 is not None:
      df2 = pd.read_csv(args.csv_raw_path1)
      df2 = drop_y_null(df2, TARGET_COL )     # Drop rows with null y values
      df2 = preprocess(df2,  args.use_genAI)
      df_processed = pd.concat([df_processed, df2], axis=0)

    if args.csv_raw_path3 == "default":
      df3 = pd.read_csv(DEFAULT_DATASET3_LOC)
      df3 = drop_y_null(df3, TARGET_COL )     # Drop rows with null y values
      df3 = preprocess(df3,  args.use_genAI)
      df_processed = pd.concat([df_processed, df3], axis=0)
    elif args.csv_raw_path3 is not None:
      df3 = pd.read_csv(args.csv_raw_path1)
      df3 = drop_y_null(df3, TARGET_COL )     # Drop rows with null y values
      df3 = preprocess(df3,  args.use_genAI)
      df_processed = pd.concat([df_processed, df3], axis=0)

    df_processed.to_csv(args.out_path+"/processed.csv", index=False)

    
    df_train_test(df_processed, args.out_path, args.test_size, args.seed)

    print(f"Saved processed, train and test CSVs to: {args.out_path}")
    print(f"Shape: {df_processed.shape}")






