import argparse
import pandas as pd

def load_data(file_path):
    with open(file_path, 'r') as f:
        headers = f.readline().strip().split()
        arities = f.readline().strip().split()
    data = pd.read_csv(file_path, sep=' ', skiprows=2, names=headers)
    return data

def compute_correlations(data):
    return data.corr()

def main():
    parser = argparse.ArgumentParser(description='Calculate correlations between variables in a dataset.')
    parser.add_argument('input_file', type=str, help='Path to the input data file')
    args = parser.parse_args()

    data = load_data(args.input_file)
    correlations = compute_correlations(data)
    print("Correlation Matrix:")
    print(correlations.round(3))

if __name__ == '__main__':
    main()
