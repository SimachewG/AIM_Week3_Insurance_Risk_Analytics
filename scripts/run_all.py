import sys
import os
import matplotlib.pyplot as plt

# Add the project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess_data
from src.exploratory_data_analysis import (
    compute_loss_ratio,
    plot_loss_ratio_by,
    univariate_analysis,
    outlier_detection,
    temporal_trend,
    vehicle_claims_analysis,
    correlation_analysis
)

# Ensure output folders exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("visualization", exist_ok=True)

def save_plot(filename):
    filepath = os.path.join("visualization", filename)
    print(f"Saving plot: {filepath}")
    return filepath

def run_all():
    # Load and preprocess
    data = load_data("data/raw/MachineLearningRating_v3.txt")
    if data is None:
        print("Data loading failed.")
        return

    data = preprocess_data(data)
    
    # Save cleaned data
    data.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Cleaned data saved to data/processed/cleaned_data.csv")

    # Compute Loss Ratio
    print("Overall Loss Ratio:", compute_loss_ratio(data))

    # Loss Ratio by categories
    for col in ['Province', 'Gender', 'VehicleType']:
        plot_loss_ratio_by(data, col)
        plt.savefig(save_plot(f"loss_ratio_by_{col.lower()}.png"))
        plt.close()

    # Univariate Analysis
    univariate_analysis(data)
    plt.savefig(save_plot("univariate_distribution.png"))
    plt.close()

    # Outlier Detection
    outlier_detection(data)
    plt.savefig(save_plot("outlier_detection.png"))
    plt.close()

    # Temporal Trends
    temporal_trend(data)
    plt.savefig(save_plot("temporal_trend.png"))
    plt.close()

    # Vehicle Claims
    vehicle_claims_analysis(data)
    plt.savefig(save_plot("vehicle_claims.png"))
    plt.close()

    # Correlation Heatmap
    correlation_analysis(data)
    plt.savefig(save_plot("correlation_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    run_all()
