import numpy as np
import pandas as pd
import subprocess
import time
import os
import csv
from datetime import datetime

def generate_mixed_data(n_samples, n_variables, discrete_ratio=0.5):
    """
    Generate a dataset with mixed continuous and discrete variables.
    
    Parameters:
    - n_samples: Number of samples (rows)
    - n_variables: Number of variables (columns)
    - discrete_ratio: Proportion of variables that should be discrete
    
    Returns:
    - DataFrame with mixed data
    """
    n_discrete = int(n_variables * discrete_ratio)
    n_continuous = n_variables - n_discrete
    
    # Generate continuous variables
    continuous_data = np.random.normal(0, 1, size=(n_samples, n_continuous))
    
    # Generate discrete variables (categorical with 3 possible values: 0, 1, 2)
    discrete_data = np.random.randint(0, 3, size=(n_samples, n_discrete))
    
    # Combine the data
    combined_data = np.hstack((continuous_data, discrete_data))
    
    # Create column names
    continuous_cols = [f'C{i+1}' for i in range(n_continuous)]
    discrete_cols = [f'D{i+1}' for i in range(n_discrete)]
    all_cols = continuous_cols + discrete_cols
    
    # Create DataFrame
    df = pd.DataFrame(combined_data, columns=all_cols)
    
    # Create metadata for GOBNILP
    metadata = {
        'continuous': continuous_cols,
        'discrete': discrete_cols,
        'n_samples': n_samples,
        'n_variables': n_variables
    }
    
    return df, metadata

def save_dataset(df, filename):
    """Save dataset in a format readable by GOBNILP"""
    df.to_csv(filename, sep=' ', index=False, header=True)
    
def run_gobnilp(dataset_path, timeout=600):
    """
    Run GOBNILP on the dataset and measure execution time
    
    Parameters:
    - dataset_path: Path to the dataset
    - timeout: Maximum execution time in seconds
    
    Returns:
    - execution_time: Time taken to run in seconds
    - status: 'success', 'timeout' or 'error'
    """
    start_time = time.time()
    
    # Add --noplot flag to prevent GOBNILP from displaying graphical output windows
    cmd = [
        "python", 
        "rungobnilp.py", 
        "--mec", 
        "--score", 
        "MixedLL", 
        "--noplot",  # Add this flag to suppress graphical output windows
        dataset_path
    ]
    
    # Create environment variables to suppress windows
    env = os.environ.copy()
    env["DISPLAY"] = ""  # Suppresses X11 display on Unix
    env["GOBNILP_NO_DISPLAY"] = "1"  # Custom variable that can be checked in GOBNILP
    
    try:
        process = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        if process.returncode == 0:
            return execution_time, 'success', process.stdout
        else:
            return execution_time, 'error', process.stderr
    
    except subprocess.TimeoutExpired:
        return timeout, 'timeout', "Process timed out"

def cleanup_test_files(test_dir, keep_results=True):
    """
    Delete test files to avoid accumulating excess files
    
    Parameters:
    - test_dir: Directory containing test files
    - keep_results: If True, keeps the results.csv file
    """
    print(f"Cleaning up test files in {test_dir}...")
    for filename in os.listdir(test_dir):
        # Skip results.csv if requested
        if keep_results and filename == "results.csv":
            continue
            
        file_path = os.path.join(test_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"  Deleted: {filename}")
        except Exception as e:
            print(f"  Error deleting {filename}: {e}")
    
    print("Cleanup completed.")

def main():
    # Parameters for test
    n_samples = 1000
    variable_counts = [3,4,5,6,7,8,9,10]  # Number of variables to test
    discrete_ratio = 0.5  # Half discrete, half continuous
    iterations = 3  # Number of times to repeat each test
    cleanup_files = True  # Set to True to delete test files after completion
    
    # Create a directory for test datasets and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"performance_test_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Setup results CSV
    results_file = os.path.join(test_dir, "results.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_variables', 'iteration', 'execution_time', 'status'])
    
    # Run tests
    for n_vars in variable_counts:
        for iteration in range(1, iterations + 1):
            print(f"Testing with {n_vars} variables (iteration {iteration}/{iterations})")
            
            # Generate dataset
            dataset, metadata = generate_mixed_data(n_samples, n_vars, discrete_ratio)
            
            # Save dataset
            dataset_filename = os.path.join(test_dir, f"data_{n_vars}vars_iter{iteration}.txt")
            save_dataset(dataset, dataset_filename)
            
            # Create metadata file to document the dataset properties
            metadata_filename = os.path.join(test_dir, f"data_{n_vars}vars_iter{iteration}_metadata.txt")
            with open(metadata_filename, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            # Run GOBNILP and record performance
            execution_time, status, output = run_gobnilp(dataset_filename)
            
            # Save output
            output_filename = os.path.join(test_dir, f"output_{n_vars}vars_iter{iteration}.txt")
            with open(output_filename, 'w') as f:
                f.write(output)
            
            # Record results
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n_vars, iteration, execution_time, status])
            
            print(f"  Time: {execution_time:.2f}s, Status: {status}")
    
    # Cleanup test files if enabled
    if cleanup_files:
        cleanup_test_files(test_dir)

if __name__ == "__main__":
    main()
