from traffic_predictor_class import TrafficPredictorCore
import os
import pandas as pd
import tensorflow as tf
import time
import pickle
import numpy as np

class TrafficPredictor(TrafficPredictorCore):
    def __init__(self):
        super().__init__()
        self.models = [
            {
                'name': 'gru_model',
                'path': 'models\\trained_models\\gru\\final_model.keras'
            },
            {
                'name': 'lstm_model',
                'path': 'models\\trained_models\\lstm\\final_model.keras'
            },
            {
                'name': 'bilstm_model',
                'path': 'models\\trained_models\\bilstm\\final_model.keras'
            },
        ]
        self.prediction_output_path = 'predicted_csv'
        os.makedirs(self.prediction_output_path, exist_ok=True)
        self.final_output_path = 'complete_oct_nov_csv'
        os.makedirs(self.final_output_path, exist_ok=True)

    def identify_october_gaps(self, df_clean):
        # Identify missing intervals in October
        october_start = pd.Timestamp("2006-10-01")
        october_end = pd.Timestamp("2006-10-31")

        # Generate all October intervals
        october_dates = pd.date_range(october_start, october_end, freq="D")
        all_october_intervals = []

        for date in october_dates:
            for interval in range(96):
                for scats in df_clean["SCATS Number"].unique():
                    all_october_intervals.append(
                        {"SCATS Number": scats, "Date": date, "interval_id": interval}
                    )

        df_october_all = pd.DataFrame(all_october_intervals)

        # Find what we actually have
        df_october_actual = df_clean[
            (df_clean["Date"] >= october_start) & (df_clean["Date"] <= october_end)
        ].copy()

        # Create keys for comparison
        df_october_all["key"] = (
            df_october_all["SCATS Number"].astype(str)
            + "_"
            + df_october_all["Date"].astype(str)
            + "_"
            + df_october_all["interval_id"].astype(str)
        )
        df_october_actual["key"] = (
            df_october_actual["SCATS Number"].astype(str)
            + "_"
            + df_october_actual["Date"].astype(str)
            + "_"
            + df_october_actual["interval_id"].astype(str)
        )

        # Find missing
        missing_keys = set(df_october_all["key"]) - set(df_october_actual["key"])
        df_october_missing = df_october_all[df_october_all["key"].isin(missing_keys)].drop(
            "key", axis=1
        )

        return df_october_missing.sort_values(["SCATS Number", "Date", "interval_id"])

    def run_predictions_for_missing(self, model_info):
        # Run predictions for a single model
        model_name = model_info['name']
        model_path = model_info['path']
        
        print(f"\n{'='*50}")
        print(f"Running predictions for model: {model_name}")
        print(f"{'='*50}\n")

        # Create model-specific output directory
        model_output_path = os.path.join(self.prediction_output_path, model_name)
        os.makedirs(model_output_path, exist_ok=True)

        print(f"Loading model from: {model_path}")
        print(f"Output will be saved to: {model_output_path}")

        # Load model
        model = tf.keras.models.load_model(model_path)

        # Initialize sequence manager
        scats_numbers = sorted(self.df_clean["SCATS Number"].unique())
        seq_manager = self.SequenceManager(scats_numbers, self.seq_length)
        seq_manager.initialize_sequences(self.df_clean)
        
        all_predictions = []

        # Phase 1: Fill October gaps (if any)
        print("Phase 1: Checking for October gaps...")
        october_gaps = self.identify_october_gaps(self.df_clean)

        if len(october_gaps) > 0:
            print(f"Found {len(october_gaps)} missing intervals in October")

            start_time = time.time()
            # Group by date and interval
            for (date, interval), group in october_gaps.groupby(["Date", "interval_id"]):
                # Predict for all SCATS Numbers in this interval
                interval_predictions = self.batch_predict_interval(
                    seq_manager,
                    group["SCATS Number"].tolist(),
                    date,
                    interval,
                    model,
                )

                # Update sequences and store predictions
                for pred in interval_predictions:
                    seq_manager.update_sequence(
                        pred["SCATS Number"],
                        pred["Date"],
                        pred["interval_id"],
                        pred["predicted_traffic"],
                    )
                    all_predictions.append(pred)

            elapsed_time = time.time() - start_time
            print(f"Total processing time: {elapsed_time:.2f} seconds")
        else:
            print("No October gaps found")

        # Phase 2: Predict November
        print("\nPhase 2: Predicting November...")
        november_start = pd.Timestamp("2006-11-01")
        november_end = pd.Timestamp("2006-11-02")
        november_dates = pd.date_range(november_start, november_end, freq="D")

        start_time = time.time()
        # Process each day in November
        for date in november_dates:
            # Process each interval
            for interval in range(96):
                # Batch predict for all SCATS Numbers
                interval_predictions = self.batch_predict_interval(
                    seq_manager, scats_numbers, date, interval, model
                )

                # Update sequences and store predictions
                for pred in interval_predictions:
                    seq_manager.update_sequence(
                        pred["SCATS Number"],
                        pred["Date"],
                        pred["interval_id"],
                        pred["predicted_traffic"],
                    )
                    all_predictions.append(pred)

        elapsed_time = time.time() - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

        # Convert predictions to DataFrame
        df_predictions = pd.DataFrame(all_predictions)

        # Save predictions
        output_file = os.path.join(model_output_path, "traffic_predictions_oct_nov.csv")
        df_predictions.to_csv(output_file, index=False)
        print(f"\nSaved predictions to: {output_file}")

        # Also save as pickle for faster loading
        pickle_file = os.path.join(model_output_path, "traffic_predictions_oct_nov.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(df_predictions, f)

        # Create summary
        summary = {
            "model_name": model_name,
            "total_predictions": len(df_predictions),
            "scats_numbers": (
                list(df_predictions["SCATS Number"].unique()) if len(df_predictions) > 0 else []
            ),
            "date_range": (
                f"{df_predictions['Date'].min()} to {df_predictions['Date'].max()}"
                if len(df_predictions) > 0
                else "No predictions"
            ),
            "prediction_stats": (
                {
                    "mean": (
                        df_predictions["predicted_traffic"].mean()
                        if len(df_predictions) > 0
                        else 0
                    ),
                    "median": (
                        df_predictions["predicted_traffic"].median()
                        if len(df_predictions) > 0
                        else 0
                    ),
                    "std": (
                        df_predictions["predicted_traffic"].std()
                        if len(df_predictions) > 0
                        else 0
                    ),
                    "min": (
                        df_predictions["predicted_traffic"].min()
                        if len(df_predictions) > 0
                        else 0
                    ),
                    "max": (
                        df_predictions["predicted_traffic"].max()
                        if len(df_predictions) > 0
                        else 0
                    ),
                }
                if len(df_predictions) > 0
                else {}
            ),
        }

        # Save summary
        summary_file = os.path.join(model_output_path, "prediction_summary.pkl")
        with open(summary_file, "wb") as f:
            pickle.dump(summary, f)

        print("\nPrediction Summary:")
        print(f"Model: {model_name}")
        print(f"Total predictions: {summary['total_predictions']}")
        print(f"SCATS Numbers processed: {len(summary['scats_numbers'])}")
        if len(df_predictions) > 0:
            print(f"Date range: {summary['date_range']}")
            print(f"Mean predicted traffic: {summary['prediction_stats']['mean']:.2f}")
            print(f"Std predicted traffic: {summary['prediction_stats']['std']:.2f}")

        print(f"\nCompleted predictions for {model_name}")
        
        # Now process the combined data
        self.process_combined_data(model_name)

    def process_combined_data(self, model_name):
        # Process and combine original data with predictions for a specific model
        print(f"\nProcessing combined data for model: {model_name}")
        
        # Filter for October and November 2006
        prediction_start = pd.Timestamp('2006-10-01')
        prediction_end = pd.Timestamp('2006-11-02 23:59:59')
        df_oct_nov = self.df_clean[
            (self.df_clean['Date'] >= prediction_start) & 
            (self.df_clean['Date'] <= prediction_end)
        ].copy()

        print(f"Original October-November data: {len(df_oct_nov)} records")

        # Load predictions for this model
        pred_file = os.path.join(self.prediction_output_path, model_name, 'traffic_predictions_oct_nov.csv')
        
        if not os.path.exists(pred_file):
            print(f"  Prediction file not found: {pred_file}")
            return
        
        df_pred = pd.read_csv(pred_file)
        df_pred['Date'] = pd.to_datetime(df_pred['Date']).dt.normalize()
        
        print(f"  Loaded {len(df_pred)} predictions")
        
        # Create a copy of October-November data for this model
        df_combined = df_oct_nov.copy()
        
        # Add a column to mark original vs predicted data
        df_combined['data_source'] = 'original'
        df_pred['data_source'] = 'predicted'
        
        # Rename predicted traffic column to match original
        df_pred = df_pred.rename(columns={'predicted_traffic': 'traffic_volume'})
        
        # Add missing columns to predictions
        print("  Adding missing columns to predictions...")
        scats_specific_cols = ['NB_LATITUDE', 'NB_LONGITUDE']
        extra_cols = [col for col in df_oct_nov.columns if col not in df_pred.columns]
        
        if extra_cols:
            # Calculate all SCATS Number mappings at once
            scats_mappings = {}
            for scats in df_combined['SCATS Number'].unique():
                scats_data = df_combined[df_combined['SCATS Number'] == scats].iloc[0]
                scats_mappings[scats] = {col: scats_data[col] for col in extra_cols if col in scats_specific_cols}
            
            # Add empty columns first
            for col in extra_cols:
                df_pred[col] = None
                
            # Fill in SCATS Number-specific columns using the mapping
            for scats in df_pred['SCATS Number'].unique():
                if scats in scats_mappings:
                    mask = df_pred['SCATS Number'] == scats
                    for col, value in scats_mappings[scats].items():
                        df_pred.loc[mask, col] = value
        
        # Ensure columns are in the same order
        df_pred = df_pred[df_combined.columns]
        
        # Combine original and predictions
        df_combined = pd.concat([df_combined, df_pred], ignore_index=True)
        
        # Remove duplicates if any
        df_combined = df_combined.drop_duplicates(subset=['SCATS Number', 'Date', 'interval_id'], keep='first')

        # If any traffic volume is negative, set to 0
        df_combined['traffic_volume'] = df_combined['traffic_volume'].clip(lower=0)

        # Identify and fill missing intervals
        print("  Identifying and filling missing intervals...")
        
        # Get unique SCATS Numbers and dates
        scats_numbers = df_combined['SCATS Number'].unique()
        dates = pd.date_range(start=prediction_start, end=prediction_end)
        
        # Pre-compute time_of_day mapping for intervals
        time_of_day_map = {i: f"{i // 4:02d}:{(i % 4) * 15:02d}" for i in range(96)}
        
        # Find missing combinations using set operations
        print("  Finding missing combinations using set operations...")
        
        # Create a set of all existing combinations
        existing_combos = set()
        for _, row in df_combined[['SCATS Number', 'Date', 'interval_id']].iterrows():
            key = (row['SCATS Number'], pd.Timestamp(row['Date']).normalize(), row['interval_id'])
            existing_combos.add(key)
        
        # Create all possible combinations
        all_combos = set()
        missing_records = []
        
        # For each SCATS Number, build a template record with all SCATS Number-specific data
        scats_templates = {}
        for scats in scats_numbers:
            # Get first record for this SCATS Number to use as template
            scats_records = df_combined[df_combined['SCATS Number'] == scats]
            if len(scats_records) > 0:
                template = scats_records.iloc[0].copy()
                scats_templates[scats] = template
                
                # Generate all combinations for this SCATS Number
                for date in dates:
                    for interval in range(96):
                        key = (scats, date, interval)
                        all_combos.add(key)
                        
                        # If this combination doesn't exist, create a record for it
                        if key not in existing_combos:
                            new_record = template.copy()
                            new_record['Date'] = date
                            new_record['interval_id'] = interval
                            new_record['time_of_day'] = time_of_day_map[interval]
                            new_record['data_source'] = 'added_missing'
                            new_record['traffic_volume'] = np.nan
                            missing_records.append(new_record)
        
        # Handle missing records
        missing_count = len(missing_records)
        print(f"  Found {missing_count} missing intervals")
        
        if missing_count > 0:
            # Create a DataFrame for missing records and calculate median values
            df_missing = pd.DataFrame(missing_records)
            
            # Calculate median values
            if len(df_missing) > 0:
                print("  Calculating interval medians...")
                median_by_date_interval = df_combined.groupby(['Date', 'interval_id'])['traffic_volume'].median()
                
                # Apply medians using vectorized operations
                print("  Applying median values to missing intervals...")
                date_interval_idx = pd.MultiIndex.from_frame(df_missing[['Date', 'interval_id']])
                df_missing['traffic_volume'] = median_by_date_interval.reindex(date_interval_idx).values
            
            # Add missing records to combined data
            df_combined = pd.concat([df_combined, df_missing], ignore_index=True)
        else:
            print("  No missing intervals found.")
            
        # Sort the final dataset
        df_combined = df_combined.sort_values(['SCATS Number', 'Date', 'interval_id'])

        # Only keep the most necessary columns
        columns_to_keep = ['SCATS Number', 'Date', 'interval_id', 'time_of_day', 'traffic_volume', 'data_source']
        df_combined = df_combined[columns_to_keep]

        # Save combined data
        output_folder = os.path.join(self.final_output_path, model_name)
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'{model_name}_complete_data.csv')
        df_combined.to_csv(output_file, index=False)

        # Save pickle file for the combined data
        pickle_file = os.path.join(output_folder, f'{model_name}_complete_data.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(df_combined, f)
        
        print(f"  Saved combined data to: {output_file}")
        print(f"  Total records: {len(df_combined)}")
        print(f"  Original records: {len(df_combined[df_combined['data_source'] == 'original'])}")
        print(f"  Predicted records: {len(df_combined[df_combined['data_source'] == 'predicted'])}")
        print(f"  Added missing records: {len(df_combined[df_combined['data_source'] == 'added_missing'])}")
        print(f"  Records with NaN traffic volume: {df_combined['traffic_volume'].isna().sum()}")

    def run_all_models(self):
        for model_info in self.models:
            self.run_predictions_for_missing(model_info)

if __name__ == "__main__":
    predictor = TrafficPredictor()
    predictor.run_all_models()