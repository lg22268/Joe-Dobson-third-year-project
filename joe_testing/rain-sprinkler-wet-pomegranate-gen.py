import argparse
import pandas as pd
from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import Categorical, ConditionalCategorical
import numpy as np

def build_model():
    rain = Categorical([[0.5, 0.5]])
    sprinkler = Categorical([[0.5, 0.5]])

    # Shape: (rain, sprinkler, wet)
    wet_probs = np.array([
        [[1.0, 0.0], [0.0, 1.0]],  # rain=0
        [[0.0, 1.0], [0.0, 1.0]],  # rain=1
    ])

    wet = ConditionalCategorical([wet_probs])

    model = BayesianNetwork(
        distributions=[rain, sprinkler, wet],
        structure=[[], [], [0, 1]]
    )

    return model


def generate_samples(model, n_samples: int):
    rain_samples = model.distributions[0].sample(n_samples)[:, 0]
    sprinkler_samples = model.distributions[1].sample(n_samples)[:, 0]

    # Combine into (n_samples, 2, 1) instead of (n_samples, 1, 2)
    parent_vals = np.stack([rain_samples, sprinkler_samples], axis=1)  # shape: (n_samples, 2)
    parent_vals = parent_vals[..., np.newaxis]  # shape: (n_samples, 2, 1)

    wet_samples = model.distributions[2].sample(n_samples, parent_vals)[:, 0]

    df = pd.DataFrame({
        "rain": rain_samples,
        "sprinkler": sprinkler_samples,
        "wet": wet_samples
    })

    return df.astype(str)


def save_samples_space_separated(df, output_file):
    with open(output_file, 'w') as f:
        f.write(' '.join(df.columns) + '\n')
        f.write(' '.join(['2'] * len(df.columns)) + '\n')
        df.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n')


def main(n=None, out=None):
    parser = argparse.ArgumentParser(description="Generate data using pomegranate >=1.0")
    parser.add_argument('-n', '--num-samples', type=int, help='Number of samples to generate')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    args = parser.parse_args() if n is None and out is None else argparse.Namespace(num_samples=n, output=out)

    model = build_model()
    df = generate_samples(model, args.num_samples)
    save_samples_space_separated(df, args.output)
    print(f"Generated {args.num_samples} samples and saved to {args.output}")


if __name__ == "__main__":
    main(1000, 'rain-sprinkler-wet.txt')
