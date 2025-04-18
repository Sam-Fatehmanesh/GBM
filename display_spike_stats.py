#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def display_spike_stats(csv_file):
    """
    Display spike statistics summary in a readable format.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file with spike statistics summary
    """
    # Load the summary data
    df = pd.read_csv(csv_file)
    
    # Convert string percentages to floats
    df['Active Cells (%)'] = df['Active Cells (%)'].str.rstrip('%').astype(float)
    
    # Convert string values to float where needed
    numeric_columns = [
        'Mean Spikes/Cell', 
        'Mean Active Neurons/Timepoint', 
        'Median Active Neurons/Timepoint',
        'Max Active Neurons/Timepoint'
    ]
    
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    # Sort by mean active neurons per timepoint
    df_sorted = df.sort_values('Mean Active Neurons/Timepoint', ascending=False)
    
    # Print a nicely formatted table
    print("\n=== SPIKE STATISTICS SUMMARY ===")
    print("\nMean and Median Active Neurons per Timepoint by Subject:")
    print("-" * 80)
    print(f"{'Subject':<10} {'Mean Active':<15} {'Median Active':<15} {'Active Cells (%)':<15} {'Mean Spikes/Cell':<15}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        subject = row['Subject']
        mean_active = row['Mean Active Neurons/Timepoint']
        median_active = row['Median Active Neurons/Timepoint']
        active_pct = row['Active Cells (%)']
        mean_spikes = row['Mean Spikes/Cell']
        
        print(f"{subject:<10} {mean_active:<15.2f} {median_active:<15.2f} {active_pct:<15.1f} {mean_spikes:<15.2f}")
    
    print("-" * 80)
    
    # Create a horizontal bar chart showing mean active neurons
    plt.figure(figsize=(12, 8))
    
    # Create bars
    plt.barh(df_sorted['Subject'], df_sorted['Mean Active Neurons/Timepoint'], color='skyblue')
    
    # Customize chart
    plt.xlabel('Mean Active Neurons per Timepoint')
    plt.ylabel('Subject')
    plt.title('Mean Number of Active Neurons per Timepoint by Subject')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add values at the end of each bar
    for i, v in enumerate(df_sorted['Mean Active Neurons/Timepoint']):
        plt.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Create a scatter plot of mean active neurons vs active cells percentage
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(
        df_sorted['Mean Active Neurons/Timepoint'], 
        df_sorted['Active Cells (%)'],
        s=100,  # Marker size
        c=df_sorted['Mean Spikes/Cell'],  # Color based on mean spikes per cell
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Mean Spikes per Cell')
    
    # Add subject labels to each point
    for i, subject in enumerate(df_sorted['Subject']):
        plt.annotate(
            subject,
            (df_sorted['Mean Active Neurons/Timepoint'].iloc[i], df_sorted['Active Cells (%)'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Customize chart
    plt.xlabel('Mean Active Neurons per Timepoint')
    plt.ylabel('Percentage of Active Cells (%)')
    plt.title('Relationship Between Neuronal Activity and Cell Activation')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Display spike statistics summary.')
    parser.add_argument('--csv_file', type=str, default='spike_statistics_report.csv',
                       help='CSV file with spike statistics summary')
    
    args = parser.parse_args()
    display_spike_stats(args.csv_file)

if __name__ == "__main__":
    main() 