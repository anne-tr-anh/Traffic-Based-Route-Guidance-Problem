import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from collections import defaultdict

class ModelEvaluator:
    def __init__(self, data_dir="oct3_csv"):
        self.data_dir = data_dir
        self.models = ["gru_model", "lstm_model", "bilstm_model"]
        self.metrics = {
            "MAE": mean_absolute_error,
            "MAPE": mean_absolute_percentage_error,
            "R2": r2_score
        }
        
    def load_data(self, model_name):
        # Load data for a specific model
        model_dir = os.path.join(self.data_dir)
        data_file = os.path.join(model_dir, f"october_1_3_combined_{model_name}.pkl")
        print("Looking for:", data_file)
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found for model {model_name}")
            
        with open(data_file, "rb") as f:
            return pickle.load(f)
    
    def prepare_evaluation_data(self):
        # Prepare data for all models
        self.model_data = {}

        for model in self.models:
            try:
                df = self.load_data(model)
                # Split into predicted and original
                pred_df = df[df["data_source"] == "predicted"].copy()
                orig_df = df[df["data_source"] == "original"].copy()

                # Merge predictions with ground truth
                merged = pd.merge(
                    pred_df,
                    orig_df[["SCATS Number", "Date", "interval_id", "time_of_day", "traffic_volume"]],
                    on=["SCATS Number", "Date", "interval_id", "time_of_day"],
                    how="left",
                    suffixes=('', '_actual')
                )
                merged = merged.rename(columns={"traffic_volume_actual": "actual_traffic"})
                # Drop rows where actual_traffic is missing
                merged = merged.dropna(subset=["actual_traffic", "traffic_volume"])
                self.model_data[model] = merged
            except Exception as e:
                print(f"Error loading data for {model}: {str(e)}")
                self.model_data[model] = None
    
    def calculate_metrics(self):
        # Calculate all metrics for all models
        self.results = defaultdict(dict)
        
        for model, df in self.model_data.items():
            if df is None or len(df) == 0:
                continue
                
            y_true = df["actual_traffic"].values
            y_pred = df["traffic_volume"].values
            
            for metric_name, metric_func in self.metrics.items():
                try:
                    score = metric_func(y_true, y_pred)
                    self.results[model][metric_name] = score
                except Exception as e:
                    print(f"Error calculating {metric_name} for {model}: {str(e)}")
                    self.results[model][metric_name] = None
    
    def generate_comparison_report(self):
        # Generate comprehensive comparison report
        if not hasattr(self, "results"):
            self.calculate_metrics()
            
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame.from_dict(self.results, orient="index")
        
        # Save results
        os.makedirs("evaluation_results", exist_ok=True)
        results_df.to_csv("evaluation_results/model_comparison.csv")
        results_df.to_markdown("evaluation_results/model_comparison.md")
        
        # Generate visualizations
        self._generate_visualizations(results_df)
        
        return results_df
    
    def _generate_visualizations(self, results_df):
        # Generate comparison visualizations
        # Metric comparison bar plot
        metrics_to_plot = ["MAE", "MAPE", "R2"]
        plt.figure(figsize=(12, 6))
        results_df[metrics_to_plot].plot(kind="bar", width=0.8)
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("evaluation_results/metric_comparison.png")
        plt.close()
        
        # Scatter plots of predictions vs actual for each model
        for model in self.models:
            if model not in self.model_data or self.model_data[model] is None:
                continue
                
            df = self.model_data[model]
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x="actual_traffic", y="traffic_volume", data=df)
            plt.plot([df["actual_traffic"].min(), df["actual_traffic"].max()], 
                    [df["actual_traffic"].min(), df["actual_traffic"].max()], 
                    'r--')
            plt.title(f"Actual vs Predicted - {model.upper()}")
            plt.xlabel("Actual Traffic Volume")
            plt.ylabel("Predicted Traffic Volume")
            plt.tight_layout()
            plt.savefig(f"evaluation_results/actual_vs_predicted_{model}.png")
            plt.close()
    
    def run_full_evaluation(self):
        # Run complete evaluation pipeline
        print("Preparing evaluation data...")
        self.prepare_evaluation_data()
        
        print("Calculating metrics...")
        self.calculate_metrics()
        
        print("Generating comparison report...")
        results = self.generate_comparison_report()
        
        print("\nEvaluation complete! Results saved in 'evaluation_results' directory.")
        return results

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()