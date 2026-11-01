"""Train a one-hot Ridge baseline and compare saved test predictions.

Run:
    python train_onehot_ridge.py --root .

Uses the official training/validation/test partitions.
Regularization is selected using validation Spearman only.

This is an internal baseline, not an exact reproduction of the
published FLIP2 implementation. Comparisons are descriptive;
statistical significance is not evaluated.

Existing output files with matching names are overwritten.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EXPECTED_COUNTS = {"train": 2403, "val": 601, "test": 702}
ALPHAS = (0.01, 0.1, 1, 10, 100, 1000)


def load_splits(root):
    parts = {}
    seen = set()

    for split, expected_count in EXPECTED_COUNTS.items():
        path = root / "official_split" / f"{split}_unseen.csv"
        frame = pd.read_csv(path)

        if not {"sequence", "target"}.issubset(frame.columns):
            raise ValueError(f"Missing sequence or target column: {path}")

        if len(frame) != expected_count:
            raise ValueError(
                f"Expected {expected_count} rows in {path}; "
                f"found {len(frame)}"
            )

        if frame.sequence.isna().any() or not frame.sequence.is_unique:
            raise ValueError(f"Missing or duplicate sequences: {path}")

        if not all(
            isinstance(sequence, str)
            and len(sequence) == 425
            and set(sequence) <= set(ALPHABET)
            for sequence in frame.sequence
        ):
            raise ValueError(f"Invalid sequence length or alphabet: {path}")

        frame["target"] = pd.to_numeric(frame.target, errors="raise")

        if not np.isfinite(frame.target.to_numpy(float)).all():
            raise ValueError(f"Non-finite targets: {path}")

        sequences = set(frame.sequence)
        if seen & sequences:
            raise ValueError(f"Sequence overlap involving {split}")

        seen.update(sequences)
        parts[split] = frame

    return parts


def encode_onehot(sequences):
    sequences = list(sequences)
    lookup = {aa: i for i, aa in enumerate(ALPHABET)}

    columns = np.array([
        [
            position * 20 + lookup[aa]
            for position, aa in enumerate(sequence)
        ]
        for sequence in sequences
    ])

    return csr_matrix(
        (
            np.ones(columns.size, dtype=np.float64),
            (
                np.repeat(np.arange(len(sequences)), 425),
                columns.ravel(),
            ),
        ),
        shape=(len(sequences), 425 * 20),
    )


def make_ridge(alpha):
    return Ridge(
        alpha=alpha,
        solver="lsqr",
        tol=1e-8,
        max_iter=10000,
    )


def load_existing_predictions(root, test):
    predictions = {}
    actual = test.target.to_numpy(float)

    for name, folder in [
        ("Transformer", "outputs_rank"),
        ("ESM_Ridge", "outputs_esm_ridge"),
        ("Hybrid", "outputs_blend"),
    ]:
        path = root / folder / "test_predictions.csv"
        saved = pd.read_csv(path)

        required = {
            "sequence",
            "experimental_fitness",
            "predicted_fitness",
        }

        if not required.issubset(saved.columns):
            raise ValueError(f"Missing prediction columns: {path}")

        if (
            not saved.sequence.is_unique
            or set(saved.sequence) != set(test.sequence)
        ):
            raise ValueError(f"Sequence membership mismatch: {path}")

        saved = saved.set_index("sequence").loc[test.sequence]

        np.testing.assert_allclose(
            saved.experimental_fitness.to_numpy(float),
            actual,
            rtol=0,
            atol=1e-7,
            err_msg=f"Experimental labels disagree: {path}",
        )

        prediction = saved.predicted_fitness.to_numpy(float)

        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite predictions: {path}")

        predictions[name] = prediction

    return predictions


def calculate_metrics(actual, predictions):
    i, j = np.triu_indices(len(actual), 1)
    true_difference = actual[i] - actual[j]
    eligible = np.abs(true_difference) >= 0.03

    high_fitness = np.zeros(len(actual), dtype=bool)
    n_high = int(np.ceil(0.10 * len(actual)))
    high_fitness[
        np.argsort(-actual, kind="stable")[:n_high]
    ] = True

    rows = []

    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite predictions: {name}")

        predicted_difference = prediction[i] - prediction[j]
        order = np.argsort(-prediction, kind="stable")

        pairwise = (
            np.mean(
                (
                    true_difference[eligible]
                    * predicted_difference[eligible] > 0
                )
                + 0.5 * (predicted_difference[eligible] == 0)
            )
            if eligible.any()
            else np.nan
        )

        rows.append({
            "Model": name,
            "MSE": np.mean((actual - prediction) ** 2),
            "MAE": np.mean(np.abs(actual - prediction)),
            "Spearman": spearmanr(actual, prediction).statistic,
            "Pairwise@0.03": pairwise,
            "Top10_hits": int(high_fitness[order[:10]].sum()),
            "Top25_hits": int(high_fitness[order[:25]].sum()),
            "Top50_hits": int(high_fitness[order[:50]].sum()),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory containing official_split and model outputs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_onehot_comparison"),
        help="Output directory; relative paths are resolved against --root",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out_dir.expanduser()

    if not out.is_absolute():
        out = root / out
    out = out.resolve()

    parts = load_splits(root)
    existing = load_existing_predictions(root, parts["test"])

    X = {
        split: encode_onehot(frame.sequence)
        for split, frame in parts.items()
    }
    y = {
        split: frame.target.to_numpy(float)
        for split, frame in parts.items()
    }

    # Select regularization using validation data only.
    trials = []

    for alpha in ALPHAS:
        candidate = make_ridge(alpha)
        candidate.fit(X["train"], y["train"])

        score = float(
            spearmanr(
                y["val"],
                candidate.predict(X["val"]),
            ).statistic
        )

        trials.append({
            "alpha": alpha,
            "validation_spearman": score,
        })

    trials = pd.DataFrame(trials)

    if not np.isfinite(trials.validation_spearman).all():
        raise ValueError("Non-finite validation Spearman")

    best_alpha = float(
        trials.loc[
            trials.validation_spearman.idxmax(),
            "alpha",
        ]
    )

    # Fit the selected model on training rows only.
    baseline = make_ridge(best_alpha)
    baseline.fit(X["train"], y["train"])

    baseline_prediction = baseline.predict(X["test"])

    predictions = {
        "OneHot_Ridge": baseline_prediction,
        **existing,
    }

    results = calculate_metrics(y["test"], predictions)

    out.mkdir(parents=True, exist_ok=True)

    trials.to_csv(out / "validation_search.csv", index=False)
    results.to_csv(out / "comparison_metrics.csv", index=False)

    pd.DataFrame({
        "sequence": parts["test"].sequence.to_numpy(),
        "experimental_fitness": y["test"],
        **predictions,
    }).to_csv(out / "matched_test_predictions.csv", index=False)

    np.savez(
        out / "onehot_ridge_head.npz",
        coef=baseline.coef_,
        intercept=baseline.intercept_,
        alpha=best_alpha,
        alphabet=ALPHABET,
    )

    configuration = {
        "baseline": "one-hot Ridge",
        "sequence_length": 425,
        "alphabet": ALPHABET,
        "feature_count": 425 * 20,
        "feature_scaling": "none",
        "target_scaling": "none",
        "alpha": best_alpha,
        "alpha_grid": list(ALPHAS),
        "selection": "validation Spearman",
        "solver": "lsqr",
        "tolerance": 1e-8,
        "max_iter": 10000,
        "training_rows": len(parts["train"]),
        "validation_rows": len(parts["val"]),
        "test_rows": len(parts["test"]),
        "pairwise_minimum_gap": 0.03,
        "high_fitness_definition": "Top ceil(10% of test N)",
        "evaluation": "Descriptive comparison; no significance testing",
        "exact_published_reproduction": False,
    }

    (out / "configuration.json").write_text(
        json.dumps(configuration, indent=2)
    )

    print("Selected Ridge alpha:", best_alpha)
    print(results.to_string(index=False))
    print("\nResults saved:", out)


if __name__ == "__main__":
    main()