import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data

def compute_loss_ratio(data):
    data['LossRatio'] = data['TotalClaims'] / data['TotalPremium']
    return data['LossRatio'].mean()

def plot_loss_ratio_by(data, column):
    if 'LossRatio' not in data.columns:
        data['LossRatio'] = data['TotalClaims'] / data['TotalPremium']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=column, y='LossRatio', data=data)
    plt.title(f'Loss Ratio by {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def univariate_analysis(data):
    numeric_cols = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

def outlier_detection(data):
    numeric_cols = ['TotalClaims', 'CustomValueEstimate', 'SumInsured']
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data[col])
        plt.title(f'Outlier Detection for {col}')
        plt.tight_layout()
        plt.show()

def temporal_trend(data):
    monthly = data.groupby('TransactionMonth').agg({
        'TotalClaims': 'sum',
        'TotalPremium': 'sum',
        'PolicyID': 'count'
    }).rename(columns={'PolicyID': 'NumPolicies'}).reset_index()

    monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']

    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    sns.lineplot(data=monthly, x='TransactionMonth', y='TotalClaims', ax=axs[0])
    axs[0].set_title('Monthly Total Claims')

    sns.lineplot(data=monthly, x='TransactionMonth', y='TotalPremium', ax=axs[1])
    axs[1].set_title('Monthly Total Premium')

    sns.lineplot(data=monthly, x='TransactionMonth', y='LossRatio', ax=axs[2])
    axs[2].set_title('Monthly Loss Ratio')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def vehicle_claims_analysis(data):
    grouped = data.groupby(['make', 'Model'])['TotalClaims'].sum().sort_values()
    print("Top 10 Models with Lowest Total Claims:\n", grouped.head(10))
    print("\nTop 10 Models with Highest Total Claims:\n", grouped.tail(10))

    # Plot
    plt.figure(figsize=(12, 6))
    grouped.tail(10).plot(kind='barh', color='coral')
    plt.title('Top 10 Vehicle Make/Models by Total Claims')
    plt.xlabel('Total Claims')
    plt.tight_layout()
    plt.show()

def correlation_analysis(data):
    plt.figure(figsize=(12, 10))
    numeric_cols = data.select_dtypes(include='number')
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_data('../data/raw/MachineLearningRating_v3.txt')
    data = preprocess_data(data)

    print("Overall Loss Ratio:", compute_loss_ratio(data))
    
    # Loss Ratio by categories
    plot_loss_ratio_by(data, 'Province')
    plot_loss_ratio_by(data, 'Gender')
    plot_loss_ratio_by(data, 'VehicleType')
    
    # Univariate + Outlier
    univariate_analysis(data)
    outlier_detection(data)

    # Temporal trend
    temporal_trend(data)

    # Vehicle analysis
    vehicle_claims_analysis(data)

    # Correlation
    correlation_analysis(data)