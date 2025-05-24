import os
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

class TrafficPredictorCore:
    def __init__(self, data_path='processed_data', seq_length=24, batch_size=128):
        self.data_path = data_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.load_data_components()

    def load_data_components(self):
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
            target_dt = target_date + timedelta(
                hours=int(target_interval // 4), minutes=int((target_interval % 4) * 15)
            )
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
                    "predicted_traffic": predicted_value,
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
                    "predicted_traffic": pred[0],
                }
            )
        return results