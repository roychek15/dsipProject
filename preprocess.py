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


    # After feature engneering
    #"Drop rows with null y values"

    df_all = drop_y_null(df_all, y_col = Y_COL)

    return df_all



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CSV")
    parser.add_argument("--csv-raw-path1", type=str, required=False, help="Path to first raw input CSV", default=DEFAULT_DATASET1_LOC)
    parser.add_argument("--csv-raw-path2", type=str, required=False, help="Path to second raw input CSV", default=DEFAULT_DATASET2_LOC)
    parser.add_argument("--out-path", type=str, default=DEFAULT_OUTPUT_LOC, help="Path to save processed CSV")

    args = parser.parse_args()

    df1 = pd.read_csv(args.csv_raw_path1)
    df2 = pd.read_csv(args.csv_raw_path2)

    df_processed = preprocess(df1, df2)
    df_processed.to_csv(args.out_path, index=False)

    print(f"Saved processed CSV to: {args.out_path}")
    print(f"Shape: {df_processed.shape}")






