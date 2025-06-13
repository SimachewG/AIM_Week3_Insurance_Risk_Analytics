import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data
#from src.data_preprocessing import load_data, preprocess_data


def compute_loss_ratio(data):
    """Calculate the overall loss ratio."""
    data['LossRatio'] = data['TotalClaims'] / data['TotalPremium']
    return data['LossRatio'].mean()


def plot_loss_ratio(data):
    """Visualize loss ratio by province with safety checks."""
    if 'LossRatio' not in data.columns:
        if 'TotalClaims' in data.columns and 'TotalPremium' in data.columns:
            data = data.copy()
            data['LossRatio'] = data['TotalClaims'] / data['TotalPremium']
            data = data.dropna(subset=['LossRatio', 'Province'])  # ensure valid values
        else:
            raise ValueError("Required columns 'TotalClaims' and 'TotalPremium' are missing.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Province', y='LossRatio', data=data)
    plt.title('Loss Ratio by Province')
    plt.xticks(rotation=45)
    plt.show()


def univariate_analysis(data):
    """Perform univariate analysis on key financial variables."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data['TotalClaims'], bins=30, kde=True)
    plt.title('Distribution of Total Claims')
    plt.show()

if __name__ == "__main__":
    data = load_data('../data/raw/MachineLearningRating_v3.txt')
    data = preprocess_data(data)
    print("Overall Loss Ratio:", compute_loss_ratio(data))
    plot_loss_ratio(data)
    univariate_analysis(data)