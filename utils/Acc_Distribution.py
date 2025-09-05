import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os
import csv
import argparse

def load_accuracy_data(file_path):
    """Load accuracy values from a CSV file."""
    accuracies = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            accuracies.append(float(row['accuracy']))
    return accuracies

def analyze_and_plot(accuracies):
    """Analyze accuracy data and generate a distribution plot."""
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample standard deviation
    n = len(accuracies)
    confidence_level = 0.95
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)  # Two-tailed t-score
    standard_error = std_acc / np.sqrt(n)
    margin_of_error = t_critical * standard_error
    confidence_interval = (mean_acc - margin_of_error, mean_acc + margin_of_error)

    # Calculate highest and lowest accuracy
    highest_acc = max(accuracies)
    lowest_acc = min(accuracies)

    # Print 95% confidence interval and additional statistics
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Margin of Error: {margin_of_error:.4f}")
    print(f"95% Confidence Interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
    print(f"Standard Deviation: {std_acc:.4f}")
    print(f"Highest Accuracy: {highest_acc:.4f}")
    print(f"Lowest Accuracy: {lowest_acc:.4f}")

    # Plot distribution of accuracy
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=30, edgecolor='black', color='skyblue')
    plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_acc:.4f}')
    plt.axvline(confidence_interval[0], color='green', linestyle='dotted', linewidth=1.5, label=f'95% CI Lower = {confidence_interval[0]:.4f}')
    plt.axvline(confidence_interval[1], color='green', linestyle='dotted', linewidth=1.5, label=f'95% CI Upper = {confidence_interval[1]:.4f}')
    plt.title(f'Distribution of Accuracy Across {n} Iterations (Range: {lowest_acc:.4f} to {highest_acc:.4f})')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('plots/accuracy_curves', exist_ok=True)
    plt.savefig('plots/accuracy_curves/accuracy_distribution.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze accuracy logs and plot distribution.")
    parser.add_argument("--csv_file", type=str, default="test_iter_all.csv", help="Path to the CSV file containing accuracy logs")
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    
    accuracies = load_accuracy_data(args.csv_file)
    analyze_and_plot(accuracies)