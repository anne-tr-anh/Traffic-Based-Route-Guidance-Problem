import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

class TrafficPredictor:
    def __init__(self):
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
        self.data_path = 'processed_data'
        self.output_path = 'predicted_csv'
        self.seq_length = 24
        self.batch_size = 128

        os.makedirs(self.output_path, exist_ok=True)

    def load_components(self):
        with open(os.path.join(self.data_path, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.data_path, "feature_cols.pkl"), "rb") as f:
            self.feature_cols = pickle.load(f)
        with open(os.path.join(self.data_path, "scats_to_idx.pkl"), "rb") as f:
            self.scats_to_idx = pickle.load(f)
        self.df_clean = pd.read_csv(os.path.join(self.data_path, "cleaned_data.csv"))
        self.df_clean["Date"] = pd.to_datetime(self.df_clean["Date"])
        self.df_clean["Date"] = self.df_clean["Date"].dt.normalize()

    class SequenceManager:
        def __init__(self, scats_numbers, seq_length):
            self.scats_numbers = scats_numbers
            self.seq_length = seq_length
            self.sequences = {scats: pd.DataFrame() for scats in scats_numbers}
            self.avg_traffic = {}

        def initialize_sequences(self, df_clean):
            for scats in self.scats_numbers:
                scats_data = df_clean[df_clean["SCATS Number"] == scats].copy()
                scats_data = scats_data.sort_values(["Date", "interval_id"])
                self.avg_traffic[scats] = (
                    scats_data.groupby("interval_id")["traffic_volume"].mean().to_dict()
                )
                self.sequences[scats] = scats_data

        def get_last_sequence(self, scats, target_date, target_interval):
            target_dt = target_date + pd.to_timedelta(target_interval * 15, unit="m")
            scats_seq = self.sequences[scats].copy()
            scats_seq["datetime"] = scats_seq["Date"] + pd.to_timedelta(
                scats_seq["interval_id"] * 15, unit="m")
            before_target = scats_seq[scats_seq["datetime"] < target_dt].sort_values("datetime")
            if len(before_target) >= self.seq_length:
                return before_target.tail(self.seq_length).drop("datetime", axis=1)
            else:
                return None

        def update_sequence(self, scats, date, interval_id, predicted_value):
            new_row = pd.DataFrame(
                [{
                    "SCATS Number": scats,
                    "Date": date,
                    "interval_id": interval_id,
                    "time_of_day": f"{interval_id//4:02d}:{(interval_id%4)*15:02d}",
                    "traffic_volume": predicted_value,
                }]
            )
            self.sequences[scats] = pd.concat(
                [self.sequences[scats], new_row], ignore_index=True)
            self.sequences[scats] = self.sequences[scats].sort_values(
                ["Date", "interval_id"])

    def engineer_features_batch(self, sequences, scats_indices, avg_traffic_dict):
        engineered_features = []
        for seq, scats_idx, avg_traffic in zip(sequences, scats_indices, avg_traffic_dict):
            seq = seq.copy()
            seq["day_of_week"] = seq["Date"].dt.dayofweek
            seq["is_weekend"] = (seq["day_of_week"] >= 5).astype(int)
            seq["dow_sin"] = np.sin(seq["day_of_week"] * (2 * np.pi / 7))
            seq["dow_cos"] = np.cos(seq["day_of_week"] * (2 * np.pi / 7))
            seq["tod_sin"] = np.sin(seq["interval_id"] * (2 * np.pi / 96))
            seq["tod_cos"] = np.cos(seq["interval_id"] * (2 * np.pi / 96))
            seq["scats_idx"] = scats_idx
            seq["traffic_lag_1"] = seq["traffic_volume"].shift(1)
            seq["traffic_lag_4"] = seq["traffic_volume"].shift(4)
            seq["traffic_lag_96"] = seq["traffic_volume"].shift(96)
            seq["avg_traffic_this_timeofday"] = seq["interval_id"].map(avg_traffic)
            seq["traffic_lag_1"] = seq["traffic_lag_1"].fillna(seq["avg_traffic_this_timeofday"])
            seq["traffic_lag_4"] = seq["traffic_lag_4"].fillna(seq["avg_traffic_this_timeofday"])
            seq["traffic_lag_96"] = seq["traffic_lag_96"].fillna(seq["avg_traffic_this_timeofday"])
            seq["days_since_prev"] = 0
            seq["after_gap"] = 0
            engineered_features.append(seq[self.feature_cols].values)
        return np.array(engineered_features)

    def prepare_batch_inputs(self, features_batch, feature_cols):
        batch_size, seq_length, n_features = features_batch.shape
        features_reshaped = features_batch.reshape(-1, n_features)
        features_scaled = self.scaler.transform(features_reshaped)
        features_scaled = features_scaled.reshape(batch_size, seq_length, n_features)
        scats_idx_position = feature_cols.index("scats_idx")
        scats_input = features_scaled[:, :, scats_idx_position].astype(int)
        feature_input = np.delete(features_scaled, scats_idx_position, axis=2)
        return [feature_input, scats_input]

    def batch_predict_interval(self, seq_manager, scats_numbers, target_date, target_interval, model):
        valid_sequences = []
        valid_scats = []
        valid_scats_indices = []
        valid_avg_traffic = []
        for scats in scats_numbers:
            seq = seq_manager.get_last_sequence(scats, target_date, target_interval)
            if seq is not None:
                valid_sequences.append(seq)
                valid_scats.append(scats)
                valid_scats_indices.append(self.scats_to_idx[scats])
                valid_avg_traffic.append(seq_manager.avg_traffic[scats])
        if not valid_sequences:
            return []
        features_batch = self.engineer_features_batch(
            valid_sequences, valid_scats_indices, valid_avg_traffic
        )
        model_inputs = self.prepare_batch_inputs(features_batch, self.feature_cols)
        predictions = model.predict(model_inputs, verbose=0, batch_size=self.batch_size)
        results = []
        for scats, pred in zip(valid_scats, predictions):
            results.append(
                {
                    "SCATS Number": scats,
                    "Date": target_date,
                    "interval_id": target_interval,
                    "time_of_day": f"{target_interval//4:02d}:{(target_interval%4)*15:02d}",
                    "traffic_volume": pred[0],
                }
            )
        return results

    def run_all_models(self):
        self.load_components()
        for model_info in self.models:
            print(f"\nRunning predictions for model: {model_info['name']}")
            model = tf.keras.models.load_model(model_info['path'])
            scats_numbers = sorted(self.df_clean["SCATS Number"].unique())
            seq_manager = self.SequenceManager(scats_numbers, self.seq_length)
            seq_manager.initialize_sequences(self.df_clean)
            all_predictions = []
            prediction_start = pd.Timestamp("2006-10-01")
            prediction_end = pd.Timestamp("2006-10-03")
            all_dates = pd.date_range(prediction_start, prediction_end, freq="D")
            for date in all_dates:
                for interval in range(96):
                    interval_predictions = self.batch_predict_interval(
                        seq_manager, scats_numbers, date, interval, model
                    )
                    for pred in interval_predictions:
                        seq_manager.update_sequence(
                            pred["SCATS Number"],
                            pred["Date"],
                            pred["interval_id"],
                            pred["traffic_volume"],
                        )
                        all_predictions.append(pred)
            df_predictions = pd.DataFrame(all_predictions)
            df_predictions['data_source'] = 'predicted'

            # Filter original data for the first 3 days of October
            df_original = self.df_clean[
                (self.df_clean["Date"] >= prediction_start) & (self.df_clean["Date"] <= prediction_end)
            ][["SCATS Number", "Date", "interval_id", "time_of_day", "traffic_volume"]].copy()
            df_original['data_source'] = 'original'

            # Keep only required columns in predictions
            df_predictions = df_predictions[["SCATS Number", "Date", "interval_id", "time_of_day", "traffic_volume", "data_source"]]

            # Combine original and predicted
            df_combined = pd.concat([df_original, df_predictions], ignore_index=True)

            # Save combined data
            combined_file = os.path.join(self.output_path, f"october_1_3_combined_{model_info['name']}.csv")
            df_combined.to_csv(combined_file, index=False)
            print(f"Saved combined data to: {combined_file}")

            # Save pickle file for the combined data
            pickle_file = os.path.join(self.output_path, f'october_1_3_combined_{model_info['name']}.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(df_combined, f)

if __name__ == "__main__":
    predictor = TrafficPredictor()
    predictor.run_all_models()