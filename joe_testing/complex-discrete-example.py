import argparse
import pandas as pd
from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import Categorical, ConditionalCategorical
import numpy as np

def build_model():
    A = Categorical([[0.5, 0.5]])
    B = Categorical([[0.5, 0.5]])
    E = Categorical([[0.5, 0.5]])

    # C depends on A and B, D depends on C and E
    # Shape: (A, B, C)
    C_probs = np.array([
        [[0.9, 0.1], [0.2, 0.8]],  # A=0
        [[0.1, 0.9], [0.7, 0.3]],  # A=1
    ])
    # Shape: (C, E, D)
    D_probs = np.array([
        [[0.8, 0.2], [0.3, 0.7]],  # C=0
        [[0.4, 0.6], [0.9, 0.1]],  # C=1
    ])

    C = ConditionalCategorical([C_probs])
    D = ConditionalCategorical([D_probs])

    model = BayesianNetwork(
        distributions=[A, B, C, D, E],
        structure=[[], [], [0, 3], [2,4], []],
    )

    return model


def generate_samples(model, n_samples: int):
    A_samples = model.distributions[0].sample(n_samples)[:, 0]
    B_samples = model.distributions[1].sample(n_samples)[:, 0]
    E_samples = model.distributions[4].sample(n_samples)[:, 0]

    # Parents for C are A and B
    C_parent_vals = np.stack([A_samples, B_samples], axis=1)[:, :, np.newaxis]
    C_samples = model.distributions[2].sample(n_samples, C_parent_vals)[:, 0]
    
    # Parents for D are C and E
    D_parent_vals = np.stack([C_samples, E_samples], axis=1)[:, :, np.newaxis]
    D_samples = model.distributions[3].sample(n_samples, D_parent_vals)[:, 0]

    df = pd.DataFrame({
        "A": A_samples,
        "B": B_samples,
        "C": C_samples,
        "D": D_samples,
        "E": E_samples
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
    main(1000, 'complex-discrete.txt')
