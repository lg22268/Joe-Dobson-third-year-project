import argparse
import pandas as pd
import numpy as np


def generate_samples(n_samples: int) -> pd.DataFrame:
    # Root node: sun_strength
    sun_strength = np.random.normal(loc=700, scale=150, size=n_samples)

    # temperature depends on sun_strength
    temperature = 0.02 * sun_strength + 10 + np.random.normal(loc=0.0, scale=2.0, size=n_samples)

    # ground_dryness depends on temperature
    ground_dryness = 0.03 * temperature + 0.2 + np.random.normal(loc=0.0, scale=0.1, size=n_samples)

    # Combine into DataFrame
    df = pd.DataFrame({
        "sun_strength": sun_strength,
        "temperature": temperature,
        "ground_dryness": ground_dryness
    })

    # Normalize each column to [0, 1]
    df = (df - df.min()) / (df.max() - df.min())
    
    return df


def save_samples_space_separated(df: pd.DataFrame, output_file: str):
    with open(output_file, 'w') as f:
        # Header row
        f.write(' '.join(df.columns) + '\n')
        # Arities: '-' for continuous
        f.write(' '.join(['-'] * len(df.columns)) + '\n')
        # Data rows
        df.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n')


def main(n=None, out=None):
    parser = argparse.ArgumentParser(description="Generate continuous-only environmental data")
    parser.add_argument('-n', '--num-samples', type=int, help='Number of samples to generate')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    args = parser.parse_args() if n is None and out is None else argparse.Namespace(num_samples=n, output=out)

    df = generate_samples(args.num_samples)
    save_samples_space_separated(df, args.output)
    print(f"Generated {args.num_samples} samples and saved to {args.output}")


if __name__ == "__main__":
    main(100, "sun-temp-dryness.txt")
