from traffic_predictor_class import TrafficPredictorCore
import os
import pandas as pd
import tensorflow as tf
import pickle

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
        self.output_path = 'predicted_csv'
        os.makedirs(self.output_path, exist_ok=True)

    def run_predictions_for_train_dates(self, model_info):
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

    def run_all_models(self):
        for model_info in self.models:
            self.run_predictions_for_train_dates(model_info)

if __name__ == "__main__":
    predictor = TrafficPredictor()
    predictor.run_all_models()