#########################################
# Data Processing and Feature Engineering
# This script loads, reshapes, cleans, and creates feature engineering traffic data, grouping by SCATS Number.
#########################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def load_and_prepare_data(file_path):

    # Load the data
    df = pd.read_csv(file_path)
    print(f"\nLoading: {file_path}...")

    # Ensure Date is in datetime type
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Explore data
    print("\n--- Data Overview ---")
    print(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Number of unique SCATS Numbers: {df['SCATS Number'].nunique()} (sites)")
    print(f"Recorded dates: from {df['Date'].dt.date.min()} to {df['Date'].dt.date.max()}")
    print(f"First 3 rows: \n{df.head(3)}")

    # Get columns that store traffic volume (V00 - V95)
    volume_cols = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]

    return df, volume_cols

def reshape_data(df, volume_cols):

    print("\nConverting to long format table...")

    new_rows = []
    for _, row in df.iterrows():
        scats_number = row["SCATS Number"]
        date = row["Date"]
        latitude = row["NB_LATITUDE"]
        longitude = row["NB_LONGITUDE"]

        for i, col in enumerate(volume_cols):
            interval_id = i
            hours = i // 4
            minutes = (i % 4) * 15
            time_of_day = f"{hours:02d}:{minutes:02d}"

            new_row = {
                "SCATS Number": scats_number,
                "Date": date,
                "NB_LATITUDE": latitude,
                "NB_LONGITUDE": longitude,
                "interval_id": interval_id, # maximum 95
                "time_of_day": time_of_day,
                "traffic_volume": row[col],
            }
            new_rows.append(new_row)

    # Save long format table
    long_df = pd.DataFrame(new_rows)
    long_df.to_csv("data/long_format_data.csv", index=False)

    print("\n--- Long Format Table Overview ---")
    print(f"Long format data shape: {long_df.shape[0]} rows, {long_df.shape[1]} columns")
    print(f"First 3 rows: \n{long_df.head(3)}")

    return long_df

def clean_unqualified_scats(df):

    print("\nIdentifying SCATS Numbers with unqualified data...")

    # Check for duplicate entries by grouping by SCATS Number and Date
    group_by_scats_date_counts = df.groupby(["SCATS Number", "Date"]).size().reset_index(name="count")
    # Traffic counts are reported every 15 minutes -> 24 hours × 4 = 96 intervals per day
    duplicate_scats_date = group_by_scats_date_counts[group_by_scats_date_counts["count"] > 96]["SCATS Number"].unique()
    print(f"\nDuplicate entries found: {len(duplicate_scats_date)}")

    # Check for SCATS Numbers with less than 5 days of data
    dates_count = df.groupby("SCATS Number")["Date"].nunique()
    insufficient_scats = dates_count[dates_count < 5].index
    print(f"SCATS Numbers with insufficient data (less than 5 days of data): {len(insufficient_scats)}")

    unqualified_scats = list(set(duplicate_scats_date) | set(insufficient_scats))
    print(f"\nUnqualified SCATS Numbers to remove: {unqualified_scats}")

    df_clean = df[~df["SCATS Number"].isin(unqualified_scats)].copy()
    print(f"Data shape after removed unqualified rows: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"{len(df) - len(df_clean)} rows have been removed ({(len(df) - len(df_clean))/len(df):.1%})")

    return df_clean

def detect_outliers(df, column):

    print("\nDetecting outliers...")
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"{len(outliers)} outliers are found in {column}")

    df[column] = df[column].clip(lower_bound, upper_bound)

    return df

def feature_engineering(df):

    print("\nFeature engineering...")

    df_features = df.copy()
    df_features = df_features.sort_values(["SCATS Number", "Date", "interval_id"])

    # Temporal features
    df_features["day_of_week"] = df_features["Date"].dt.dayofweek
    df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)
    df_features["dow_sin"] = np.sin(df_features["day_of_week"] * (2 * np.pi / 7))
    df_features["dow_cos"] = np.cos(df_features["day_of_week"] * (2 * np.pi / 7))
    df_features["tod_sin"] = np.sin(df_features["interval_id"] * (2 * np.pi / 96))
    df_features["tod_cos"] = np.cos(df_features["interval_id"] * (2 * np.pi / 96))

    # Gap handling features
    df_features["days_since_prev"] = df_features.groupby("SCATS Number")["Date"].diff().dt.days.fillna(0)
    df_features["after_gap"] = (df_features["days_since_prev"] > 1).astype(int)

    # SCATS Number encodings
    scats_numbers = df_features["SCATS Number"].unique()
    scats_to_idx = {scats: idx for idx, scats in enumerate(scats_numbers)}
    df_features["scats_idx"] = df_features["SCATS Number"].map(scats_to_idx)

    # Lag features grouped by SCATS Number
    grouped = df_features.groupby("SCATS Number")
    df_features["traffic_lag_1"] = grouped["traffic_volume"].shift(1)
    df_features["traffic_lag_4"] = grouped["traffic_volume"].shift(4)
    df_features["traffic_lag_96"] = grouped["traffic_volume"].shift(96)
    time_means = df_features.groupby(["SCATS Number", "interval_id"])["traffic_volume"].transform("mean")
    df_features["avg_traffic_this_timeofday"] = time_means

    lag_cols = ["traffic_lag_1", "traffic_lag_4", "traffic_lag_96"]
    for col in lag_cols:
        df_features[col] = df_features[col].fillna(df_features["avg_traffic_this_timeofday"])

    return df_features, scats_to_idx

def create_sequences(df, seq_length=24):

    print(f"\nCreating sequences of length {seq_length} for every SCATS Numbers...")

    feature_cols = [
        "traffic_volume", "dow_sin", "dow_cos", "tod_sin", "tod_cos",
        "is_weekend", "after_gap", "days_since_prev", "scats_idx",
        "traffic_lag_1", "traffic_lag_4", "traffic_lag_96", "avg_traffic_this_timeofday"
    ]

    X_sequences, y_targets, metadata = [], [], []
    scats_groups = df.groupby("SCATS Number")

    for scats_num, scats_df in scats_groups:
        scats_df = scats_df.sort_values(["Date", "interval_id"])
        segment_id = (scats_df["days_since_prev"] > 1).cumsum()

        for segment, segment_df in scats_df.groupby(segment_id):
            if len(segment_df) < seq_length + 1:
                continue

            for i in range(len(segment_df) - seq_length):
                X_seq = segment_df.iloc[i:i + seq_length][feature_cols].values
                y_target = segment_df.iloc[i + seq_length]["traffic_volume"]
                X_sequences.append(X_seq)
                y_targets.append(y_target)

                meta = {
                    "SCATS Number": scats_num,
                    "target_date": segment_df.iloc[i + seq_length]["Date"],
                    "target_interval": segment_df.iloc[i + seq_length]["interval_id"],
                    "target_time": segment_df.iloc[i + seq_length]["time_of_day"],
                }
                metadata.append(meta)

    X = np.array(X_sequences)
    y = np.array(y_targets)
    metadata_df = pd.DataFrame(metadata)
    print(f"Created {len(X)} sequences")

    return X, y, metadata_df, feature_cols

def split_data(X, y, metadata_df, test_ratio=0.2):

    print(f"\nSplitting sequences with test ratio of {test_ratio}...")
    unique_dates = pd.to_datetime(metadata_df["target_date"]).dt.date.unique()
    unique_dates = np.sort(unique_dates)
    split_idx = int(len(unique_dates) * (1 - test_ratio))
    split_date = unique_dates[split_idx]
    print(f"\nSplit date: {split_date}")

    metadata_df["target_date_obj"] = pd.to_datetime(metadata_df["target_date"]).dt.date
    train_mask = metadata_df["target_date_obj"] < split_date
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_train = metadata_df[train_mask].reset_index(drop=True)
    meta_test = metadata_df[test_mask].reset_index(drop=True)

    print(f"\nSequences for training: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test)):.1%})")
    print(f"Sequences for testing: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test)):.1%})")

    train_scats = set(meta_train["SCATS Number"])
    test_scats = set(meta_test["SCATS Number"])
    if test_scats - train_scats:
        print(f"\nWarning: {len(test_scats - train_scats)} SCATS Numbers have no training data")
    if train_scats - test_scats:
        print(f"\nWarning: {len(train_scats - test_scats)} SCATS Numbers have no test data")

    return X_train, X_test, y_train, y_test, meta_train, meta_test

def normalise_data(X_train, X_test):

    print("\nNormalising data...")
    n_features = X_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

    return X_train_scaled, X_test_scaled, scaler

def prepare_inputs(X, feature_cols):

    scats_idx = feature_cols.index("scats_idx")
    scats_input = X[:, :, scats_idx].astype(int)
    feature_indices = [i for i in range(X.shape[2]) if i != scats_idx]
    feature_input = X[:, :, feature_indices]

    return [feature_input, scats_input]

def save_processed_data(processed_data, output_dir="processed_data"):

    print(f"\nSaving processed data into {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(os.path.join(output_dir, "X_train.npz"), data=processed_data["X_train"])
    np.savez_compressed(os.path.join(output_dir, "X_test.npz"), data=processed_data["X_test"])
    np.savez_compressed(os.path.join(output_dir, "y_train.npz"), data=processed_data["y_train"])
    np.savez_compressed(os.path.join(output_dir, "y_test.npz"), data=processed_data["y_test"])
    processed_data["meta_train"].to_csv(os.path.join(output_dir, "meta_train.csv"), index=False)
    processed_data["meta_test"].to_csv(os.path.join(output_dir, "meta_test.csv"), index=False)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(processed_data["scaler"], f)
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(processed_data["feature_cols"], f)
    with open(os.path.join(output_dir, "scats_to_idx.pkl"), "wb") as f:
        pickle.dump(processed_data["scats_to_idx"], f)

    print("\nSaved successfully!")

def load_processed_data(input_dir="processed_data"):

    print(f"Loading processed data from {input_dir}...")
    X_train = np.load(os.path.join(input_dir, "X_train.npz"))["data"]
    X_test = np.load(os.path.join(input_dir, "X_test.npz"))["data"]
    y_train = np.load(os.path.join(input_dir, "y_train.npz"))["data"]
    y_test = np.load(os.path.join(input_dir, "y_test.npz"))["data"]
    meta_train = pd.read_csv(os.path.join(input_dir, "meta_train.csv"))
    meta_test = pd.read_csv(os.path.join(input_dir, "meta_test.csv"))

    with open(os.path.join(input_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(input_dir, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(input_dir, "scats_to_idx.pkl"), "rb") as f:
        scats_to_idx = pickle.load(f)

    X_train_inputs = prepare_inputs(X_train, feature_cols)
    X_test_inputs = prepare_inputs(X_test, feature_cols)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_inputs": X_train_inputs,
        "X_test_inputs": X_test_inputs,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "scats_to_idx": scats_to_idx,
        "n_scats": len(scats_to_idx),
        "n_features": X_train.shape[2] - 1,
    }

def process_data(file_path, output_dir="processed_data"):

    print("\nData processing starts:")

    os.makedirs(output_dir, exist_ok=True)

    df, volume_cols = load_and_prepare_data(file_path)
    df_long = reshape_data(df, volume_cols)
    df_clean = clean_unqualified_scats(df_long)
    df_clean = detect_outliers(df_clean, "traffic_volume")
    df_clean["traffic_volume"] = df_clean.groupby(["SCATS Number", "interval_id"])["traffic_volume"]\
        .transform(lambda x: x.fillna(x.median()))
    df_clean.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)

    print("\nData processing completed successfully!")

    return df_clean

def run_feature_engineering(df, seq_length=24, test_ratio=0.2, output_dir="processed_data"):

    print("\nFeature engineering starts:")

    df_features, scats_to_idx = feature_engineering(df)
    X, y, metadata_df, feature_cols = create_sequences(df_features, seq_length)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_data(X, y, metadata_df, test_ratio)
    X_train_scaled, X_test_scaled, scaler = normalise_data(X_train, X_test)
    X_train_inputs = prepare_inputs(X_train_scaled, feature_cols)
    X_test_inputs = prepare_inputs(X_test_scaled, feature_cols)

    processed_data = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_inputs": X_train_inputs,
        "X_test_inputs": X_test_inputs,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "scats_to_idx": scats_to_idx,
        "n_scats": len(scats_to_idx),
        "n_features": X_train_scaled.shape[2] - 1,
    }

    save_processed_data(processed_data, output_dir)

    print("\nFeature engineering completed successfully!")

    return processed_data

if __name__ == "__main__":

    file_path = "data\\Scats_Data_October_2006.xlsx" # input data file
    df = pd.read_excel(file_path, sheet_name="Data")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Making aggregation rules
    v_columns = [col for col in df.columns if col.startswith('V')]
    non_v_columns = [col for col in df.columns if col not in v_columns and col not in ['SCATS Number', 'Date']]
    agg_dict = {col: 'mean' for col in v_columns}
    agg_dict.update({col: 'first' for col in non_v_columns})

    # Grouping the data by SCATS Number and Date with aggregation rules applied
    grouped_df = df.groupby(['SCATS Number', 'Date']).agg(agg_dict).reset_index()
    grouped_df[v_columns] = np.ceil(grouped_df[v_columns])
    grouped_df.to_csv(os.path.join("data", "grouped_scats_data.csv"), index=False)

    # Processing and feature engineering
    output_dir = "processed_data" # output folder
    df_clean = process_data(file_path="data\\grouped_scats_data.csv", output_dir=output_dir)
    df_clean["Date"] = pd.to_datetime(df_clean["Date"]).dt.normalize()
    processed_data = run_feature_engineering(df_clean, seq_length=24, test_ratio=0.2, output_dir=output_dir)