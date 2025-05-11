import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl

def setup_publication_style():
    """Configure matplotlib for publication-quality figures"""
    # Use serif fonts for a more scientific publication look
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })
    
    # Higher DPI for better quality in Word
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

def load_results(file_path):
    """Load results from CSV file"""
    df = pd.read_csv(file_path)
    return df

def analyze_results(df):
    """Analyze results and prepare for plotting"""
    # Group by number of variables and calculate mean and std of execution time
    grouped = df.groupby('num_variables')
    mean_times = grouped['execution_time'].mean()
    std_times = grouped['execution_time'].std()
    
    return mean_times, std_times

def plot_performance_line(df, output_dir, file_formats=None):
    """Create a publication-quality line plot showing performance vs number of variables"""
    if file_formats is None:
        file_formats = ['png', 'pdf', 'svg']
    
    mean_times, std_times = analyze_results(df)
    x = mean_times.index
    y = mean_times.values
    y_err = std_times.values
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create line plot with error bars
    line = ax.errorbar(x, y, yerr=y_err, fmt='-o', capsize=4, 
                      color='#1f77b4', ecolor='#1f77b4', 
                      elinewidth=1, capthick=1, 
                      markersize=6, markerfacecolor='white')
    
    # Add trend line (polynomial fit)
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    x_trend = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_trend, p(x_trend), '--', color='#ff7f0e', 
            linewidth=1.5, label='Trend (cubic fit)')
    
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('GOBNILP Performance vs. Number of Variables')
    
    # Use logarithmic scale for y-axis to better visualize growth
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve x-axis ticks
    ax.set_xticks(x)
    
    # Add legend
    ax.legend(['Execution Time ± Std Dev', 'Trend (cubic fit)'])
    
    # Add annotations showing values
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(f'{yi:.1f}s', 
                   xy=(xi, yi), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center', 
                   fontsize=9)
    
    plt.tight_layout()
    
    # Save figure in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.join(output_dir, 'performance_line')
    
    for fmt in file_formats:
        plt.savefig(f'{base_filename}.{fmt}', format=fmt, bbox_inches='tight')
    
    plt.close()

def plot_performance_bar(df, output_dir, file_formats=None):
    """Create a publication-quality bar chart showing performance vs number of variables"""
    if file_formats is None:
        file_formats = ['png', 'pdf', 'svg']
    
    mean_times, std_times = analyze_results(df)
    x = mean_times.index
    y = mean_times.values
    y_err = std_times.values
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bar plot with error bars
    bars = ax.bar(x, y, yerr=y_err, 
                 color='#1f77b4', alpha=0.7,
                 capsize=4, ecolor='black', 
                 error_kw={'elinewidth': 1, 'capthick': 1})
    
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('GOBNILP Performance vs. Number of Variables')
    
    # Improve x-axis ticks
    ax.set_xticks(x)
    
    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', 
                   fontsize=9)
    
    plt.tight_layout()
    
    # Save figure in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.join(output_dir, 'performance_bar')
    
    for fmt in file_formats:
        plt.savefig(f'{base_filename}.{fmt}', format=fmt, bbox_inches='tight')
    
    plt.close()

def plot_performance_scatter(df, output_dir, file_formats=None):
    """Create a publication-quality scatter plot showing all data points"""
    if file_formats is None:
        file_formats = ['png', 'pdf', 'svg']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create scatter plot for all data points
    scatter = ax.scatter(df['num_variables'], df['execution_time'], 
                        c=df['num_variables'], cmap='viridis',
                        alpha=0.7, s=50, edgecolors='black')
    
    # Add trend line (polynomial fit)
    x = df['num_variables']
    y = df['execution_time']
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    x_trend = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_trend, p(x_trend), '--', color='red', 
            linewidth=2, label='Trend (cubic fit)')
    
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Mixed log-likelihood score performance')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve x-axis ticks
    ax.set_xticks(sorted(df['num_variables'].unique()))
    
    # Add legend
    ax.legend(['Individual Runs', 'Trend'])
    
    plt.tight_layout()
    
    # Save figure in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.join(output_dir, 'performance_scatter')
    
    for fmt in file_formats:
        plt.savefig(f'{base_filename}.{fmt}', format=fmt, bbox_inches='tight')
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize GOBNILP performance test results')
    parser.add_argument('--input', default='performance_test_*/results.csv', 
                        help='Path to results CSV file (supports glob patterns)')
    parser.add_argument('--output', default='performance_charts', 
                        help='Directory to save output charts')
    parser.add_argument('--formats', default='png,pdf,svg', 
                        help='Comma-separated list of output file formats')
    
    args = parser.parse_args()
    file_formats = args.formats.split(',')
    
    # Try to find the results file if using default pattern
    if '*' in args.input:
        import glob
        result_files = sorted(glob.glob(args.input))
        if not result_files:
            print(f"No files found matching pattern: {args.input}")
            return
        input_file = result_files[-1]  # Use the most recent results file
        print(f"Using most recent results file: {input_file}")
    else:
        input_file = args.input
        
    # Setup publication style
    setup_publication_style()
    
    # Load results
    try:
        results_df = load_results(input_file)
        print(f"Loaded {len(results_df)} rows from {input_file}")
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Generate plots
    plot_performance_line(results_df, args.output, file_formats)
    plot_performance_bar(results_df, args.output, file_formats)
    plot_performance_scatter(results_df, args.output, file_formats)
    
    print(f"Charts saved to {args.output} directory in formats: {', '.join(file_formats)}")
    print("These charts are ready for inclusion in your scientific report.")

if __name__ == "__main__":
    main()
