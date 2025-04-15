#!/usr/bin/env python
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from tqdm import tqdm

def analyze_subject_spikes(h5_file):
    """
    Analyze spike statistics for a single subject.
    
    Parameters
    ----------
    h5_file : str
        Path to the processed h5 file containing spike data
    
    Returns
    -------
    dict
        Dictionary containing spike statistics for the subject
    """
    subject_name = os.path.basename(h5_file).split('_spikes')[0]
    print(f"Analyzing spike statistics for {subject_name}")
    
    with h5py.File(h5_file, 'r') as f:
        spikes = f['spikes'][:]
        num_timepoints, num_cells = spikes.shape
        
        # Get basic spike statistics
        total_spikes = np.sum(spikes > 0)
        active_cells = np.sum(np.sum(spikes > 0, axis=0) > 0)
        
        # Compute temporal spike statistics (for each time point)
        spikes_per_timepoint = np.sum(spikes > 0, axis=1)  # Count of cells spiking at each timepoint
        mean_spikes_per_timepoint = np.mean(spikes_per_timepoint)
        median_spikes_per_timepoint = np.median(spikes_per_timepoint)
        max_spikes_per_timepoint = np.max(spikes_per_timepoint)
        
        # Compute per-cell spike statistics
        spikes_per_cell = np.sum(spikes > 0, axis=0)  # Count of spikes for each cell
        mean_spikes_per_cell = np.mean(spikes_per_cell)
        median_spikes_per_cell = np.median(spikes_per_cell)
        max_spikes_per_cell = np.max(spikes_per_cell)
        
        # Compute spike amplitude statistics
        nonzero_spikes = spikes[spikes > 0]
        mean_spike_amplitude = np.mean(nonzero_spikes) if len(nonzero_spikes) > 0 else 0
        median_spike_amplitude = np.median(nonzero_spikes) if len(nonzero_spikes) > 0 else 0
        max_spike_amplitude = np.max(nonzero_spikes) if len(nonzero_spikes) > 0 else 0
        
        # Compute temporal distribution of spike events
        # (compute mean and median number of neurons active at each time point)
        timepoint_stats = {
            'mean_active_neurons': mean_spikes_per_timepoint,
            'median_active_neurons': median_spikes_per_timepoint,
            'max_active_neurons': max_spikes_per_timepoint,
            'timepoints_with_activity': np.sum(spikes_per_timepoint > 0),
            'percent_timepoints_with_activity': 100 * np.sum(spikes_per_timepoint > 0) / num_timepoints,
            'temporal_activity': spikes_per_timepoint
        }
        
        # Compute cell-specific spike statistics
        cell_stats = {
            'mean_spikes_per_cell': mean_spikes_per_cell,
            'median_spikes_per_cell': median_spikes_per_cell,
            'max_spikes_per_cell': max_spikes_per_cell,
            'active_cells': active_cells,
            'percent_active_cells': 100 * active_cells / num_cells,
            'spikes_per_cell': spikes_per_cell
        }
        
        # Compute spike amplitude statistics
        amplitude_stats = {
            'mean_amplitude': mean_spike_amplitude,
            'median_amplitude': median_spike_amplitude,
            'max_amplitude': max_spike_amplitude
        }
        
        return {
            'subject': subject_name,
            'num_timepoints': num_timepoints,
            'num_cells': num_cells,
            'total_spikes': total_spikes,
            'timepoint_stats': timepoint_stats,
            'cell_stats': cell_stats,
            'amplitude_stats': amplitude_stats
        }

def create_statistics_report(stats_list, output_file):
    """
    Create a PDF report with spike statistics for all subjects.
    
    Parameters
    ----------
    stats_list : list
        List of dictionaries containing spike statistics for each subject
    output_file : str
        Path to save the output PDF file
    """
    print(f"Creating statistics report: {output_file}")
    
    with PdfPages(output_file) as pdf:
        # Create a summary table for all subjects
        summary_data = {
            'Subject': [],
            'Cells': [],
            'Timepoints': [],
            'Active Cells (%)': [],
            'Mean Spikes/Cell': [],
            'Mean Active Neurons/Timepoint': [],
            'Median Active Neurons/Timepoint': [],
            'Max Active Neurons/Timepoint': []
        }
        
        for stats in stats_list:
            subject = stats['subject']
            summary_data['Subject'].append(subject)
            summary_data['Cells'].append(stats['num_cells'])
            summary_data['Timepoints'].append(stats['num_timepoints'])
            summary_data['Active Cells (%)'].append(f"{stats['cell_stats']['percent_active_cells']:.1f}%")
            summary_data['Mean Spikes/Cell'].append(f"{stats['cell_stats']['mean_spikes_per_cell']:.2f}")
            summary_data['Mean Active Neurons/Timepoint'].append(f"{stats['timepoint_stats']['mean_active_neurons']:.2f}")
            summary_data['Median Active Neurons/Timepoint'].append(f"{stats['timepoint_stats']['median_active_neurons']:.2f}")
            summary_data['Max Active Neurons/Timepoint'].append(f"{stats['timepoint_stats']['max_active_neurons']:.0f}")
        
        # Create summary table figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Add the table
        df = pd.DataFrame(summary_data)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.title('Spike Statistics Summary Across Subjects', fontsize=16, pad=20)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Create a figure showing mean and median active neurons per timepoint for each subject
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Sort subjects by mean active neurons
        sorted_indices = np.argsort([s['timepoint_stats']['mean_active_neurons'] for s in stats_list])
        subjects = [stats_list[i]['subject'] for i in sorted_indices]
        mean_active = [stats_list[i]['timepoint_stats']['mean_active_neurons'] for i in sorted_indices]
        median_active = [stats_list[i]['timepoint_stats']['median_active_neurons'] for i in sorted_indices]
        
        # Plot mean active neurons
        ax1.bar(subjects, mean_active, color='skyblue')
        ax1.set_title('Mean Number of Active Neurons per Timepoint', fontsize=14)
        ax1.set_ylabel('Mean Active Neurons')
        ax1.set_xticklabels(subjects, rotation=45, ha='right')
        
        # Plot median active neurons
        ax2.bar(subjects, median_active, color='lightgreen')
        ax2.set_title('Median Number of Active Neurons per Timepoint', fontsize=14)
        ax2.set_ylabel('Median Active Neurons')
        ax2.set_xticklabels(subjects, rotation=45, ha='right')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Create individual subject pages
        for stats in stats_list:
            subject = stats['subject']
            
            # Create a multi-panel figure for this subject
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            plt.suptitle(f"Spike Statistics: {subject}", fontsize=16)
            
            # Plot 1: Histogram of spikes per timepoint
            axs[0, 0].hist(stats['timepoint_stats']['temporal_activity'], bins=30, color='skyblue', edgecolor='black')
            axs[0, 0].set_title('Distribution of Active Neurons per Timepoint')
            axs[0, 0].set_xlabel('Number of Active Neurons')
            axs[0, 0].set_ylabel('Count (Timepoints)')
            axs[0, 0].axvline(stats['timepoint_stats']['mean_active_neurons'], color='red', linestyle='--', 
                             label=f"Mean: {stats['timepoint_stats']['mean_active_neurons']:.2f}")
            axs[0, 0].axvline(stats['timepoint_stats']['median_active_neurons'], color='green', linestyle='--', 
                             label=f"Median: {stats['timepoint_stats']['median_active_neurons']:.2f}")
            axs[0, 0].legend()
            
            # Plot 2: Histogram of spikes per cell
            axs[0, 1].hist(stats['cell_stats']['spikes_per_cell'], bins=30, color='lightgreen', edgecolor='black')
            axs[0, 1].set_title('Distribution of Spikes per Cell')
            axs[0, 1].set_xlabel('Number of Spikes')
            axs[0, 1].set_ylabel('Count (Cells)')
            axs[0, 1].axvline(stats['cell_stats']['mean_spikes_per_cell'], color='red', linestyle='--', 
                             label=f"Mean: {stats['cell_stats']['mean_spikes_per_cell']:.2f}")
            axs[0, 1].axvline(stats['cell_stats']['median_spikes_per_cell'], color='green', linestyle='--', 
                             label=f"Median: {stats['cell_stats']['median_spikes_per_cell']:.2f}")
            axs[0, 1].legend()
            
            # Plot 3: Summary statistics as text
            axs[1, 0].axis('off')
            stats_text = (
                f"Total Timepoints: {stats['num_timepoints']}\n"
                f"Total Cells: {stats['num_cells']}\n"
                f"Total Spike Events: {stats['total_spikes']}\n\n"
                f"Active Cells: {stats['cell_stats']['active_cells']} ({stats['cell_stats']['percent_active_cells']:.1f}%)\n"
                f"Timepoints with Activity: {stats['timepoint_stats']['timepoints_with_activity']} "
                f"({stats['timepoint_stats']['percent_timepoints_with_activity']:.1f}%)\n\n"
                f"Mean Spikes per Cell: {stats['cell_stats']['mean_spikes_per_cell']:.2f}\n"
                f"Median Spikes per Cell: {stats['cell_stats']['median_spikes_per_cell']:.2f}\n"
                f"Max Spikes per Cell: {stats['cell_stats']['max_spikes_per_cell']:.0f}\n\n"
                f"Mean Active Neurons per Timepoint: {stats['timepoint_stats']['mean_active_neurons']:.2f}\n"
                f"Median Active Neurons per Timepoint: {stats['timepoint_stats']['median_active_neurons']:.2f}\n"
                f"Max Active Neurons per Timepoint: {stats['timepoint_stats']['max_active_neurons']:.0f}\n\n"
                f"Mean Spike Amplitude: {stats['amplitude_stats']['mean_amplitude']:.4f}\n"
                f"Median Spike Amplitude: {stats['amplitude_stats']['median_amplitude']:.4f}\n"
                f"Max Spike Amplitude: {stats['amplitude_stats']['max_amplitude']:.4f}"
            )
            axs[1, 0].text(0.05, 0.95, stats_text, fontsize=10, va='top')
            
            # Plot 4: Temporal activity pattern (smoothed)
            window_size = min(100, stats['num_timepoints'] // 100)  # Adaptive window size 
            if window_size > 0:
                smoothed_activity = np.convolve(stats['timepoint_stats']['temporal_activity'], 
                                               np.ones(window_size)/window_size, mode='valid')
                time_indices = np.arange(len(smoothed_activity))
                axs[1, 1].plot(time_indices, smoothed_activity, color='darkblue')
                axs[1, 1].set_title(f'Temporal Pattern of Neural Activity (Smoothed, window={window_size})')
                axs[1, 1].set_xlabel('Time (frames)')
                axs[1, 1].set_ylabel('Active Neurons')
                # Add line at the mean
                axs[1, 1].axhline(stats['timepoint_stats']['mean_active_neurons'], color='red', linestyle='--', 
                                 label=f"Mean: {stats['timepoint_stats']['mean_active_neurons']:.2f}")
                axs[1, 1].legend()
            else:
                axs[1, 1].text(0.5, 0.5, "Insufficient data for smoothing", ha='center', va='center', fontsize=12)
                axs[1, 1].set_title('Temporal Pattern of Neural Activity')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"Statistics report saved to: {output_file}")
    
    # Create a CSV file with the summary data for further analysis
    csv_file = output_file.replace('.pdf', '.csv')
    pd.DataFrame(summary_data).to_csv(csv_file, index=False)
    print(f"Summary statistics saved to: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze spike statistics from processed spike data.')
    parser.add_argument('--data_dir', type=str, default='spike_processed_data_2018',
                       help='Directory containing processed spike data files')
    parser.add_argument('--output_file', type=str, default='spike_statistics_report.pdf',
                       help='Output PDF file for the statistics report')
    
    args = parser.parse_args()
    
    # Find all h5 files with spike data
    spike_files = []
    for file in os.listdir(args.data_dir):
        if file.endswith('_spikes.h5') or file.endswith('_spikes_built_in.h5'):
            spike_files.append(os.path.join(args.data_dir, file))
    
    spike_files.sort()
    print(f"Found {len(spike_files)} spike data files")
    
    # Analyze each file
    stats_list = []
    for spike_file in tqdm(spike_files, desc="Analyzing files"):
        try:
            stats = analyze_subject_spikes(spike_file)
            stats_list.append(stats)
        except Exception as e:
            print(f"Error analyzing {spike_file}: {str(e)}")
    
    # Generate statistics report
    if stats_list:
        output_file = args.output_file
        create_statistics_report(stats_list, output_file)
    else:
        print("No valid statistics to report")
    
if __name__ == "__main__":
    main() 