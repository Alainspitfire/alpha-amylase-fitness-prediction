
# bootstrap_model_comparison.py

"""Paired bootstrap comparison of three saved model predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PREDICTIONS = {
    "Transformer": "transformer_prediction",
    "ESM Ridge": "esm_ridge_prediction",
    "Hybrid": "hybrid_prediction",
}


def calculate_metrics(y, pred):
    i, j = np.triu_indices(len(y), k=1)

    actual_diff = y[i] - y[j]
    pred_diff = pred[i] - pred[j]

    valid = (
        (np.abs(actual_diff) >= 0.03)
        & (actual_diff != 0)
    )

    if valid.any():
        products = actual_diff[valid] * pred_diff[valid]

        correct = np.count_nonzero(products > 0)
        ties = np.count_nonzero(products == 0)

        pairwise = (
            correct + 0.5 * ties
        ) / valid.sum()
    else:
        pairwise = float("nan")

    return np.array([
        np.mean(np.abs(y - pred)),
        np.mean((y - pred) ** 2),
        float(spearmanr(y, pred).statistic),
        pairwise,
    ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs_official_test/official_test_predictions.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_official_test/bootstrap"),
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.replicates < 20:
        raise ValueError(
            "Use at least 20 bootstrap replicates"
        )

    df = pd.read_csv(args.input)

    required = [
        "sequence",
        "experimental_fitness",
        *PREDICTIONS.values(),
    ]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["sequence"].duplicated().any():
        raise ValueError("Duplicate sequences")

    if df[required].isna().any().any():
        raise ValueError("Missing prediction values")

    values = df[required[1:]].to_numpy(float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite values")

    y = df["experimental_fitness"].to_numpy(float)

    predictions = {
        name: df[column].to_numpy(float)
        for name, column in PREDICTIONS.items()
    }

    metric_names = [
        "MAE",
        "MSE",
        "Spearman",
        "Pairwise@0.03",
    ]

    comparisons = [
        "Transformer",
        "ESM Ridge",
    ]

    original_metrics = {
        name: calculate_metrics(y, pred)
        for name, pred in predictions.items()
    }

    draws = {
        name: []
        for name in comparisons
    }

    rng = np.random.default_rng(args.seed)

    for b in range(args.replicates):
        # Sample proteins, preserving alignment
        # across all three model predictions.
        indices = rng.integers(
            low=0,
            high=len(y),
            size=len(y),
        )

        bootstrap_metrics = {
            name: calculate_metrics(
                y[indices],
                pred[indices],
            )
            for name, pred in predictions.items()
        }

        for name in comparisons:
            difference = (
                bootstrap_metrics["Hybrid"]
                - bootstrap_metrics[name]
            )

            draws[name].append(difference)

        if (b + 1) % 100 == 0:
            print(
                f"Bootstrap {b + 1}/{args.replicates}",
                flush=True,
            )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for name in comparisons:
        samples = np.stack(draws[name])

        for index, metric in enumerate(metric_names):
            values = samples[:, index]
            values = values[np.isfinite(values)]

            rows.append({
                "comparison": f"Hybrid - {name}",
                "metric": metric,
                "observed_difference": (
                    original_metrics["Hybrid"][index]
                    - original_metrics[name][index]
                ),
                "ci_lower_95": float(
                    np.quantile(values, 0.025)
                ),
                "ci_upper_95": float(
                    np.quantile(values, 0.975)
                ),
                "bootstrap_replicates": len(values),
            })

    results = pd.DataFrame(rows)

    results.to_csv(
        args.out_dir / "paired_bootstrap_intervals.csv",
        index=False,
    )

    print("\n===== PAIRED BOOTSTRAP RESULTS =====")
    print(results.round(6).to_string(index=False))

    print("\nInterpretation:")
    print("Lower MAE/MSE is better.")
    print("Higher Spearman/pairwise is better.")
    print(
        "Bootstrap intervals are exploratory; "
        "related protein variants may not be independent."
    )


if __name__ == "__main__":
    main()
