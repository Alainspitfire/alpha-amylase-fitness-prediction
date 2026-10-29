
"""Evaluate official-split-trained models using saved predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


MODELS = {
    "Transformer": "transformer_prediction",
    "ESM_Ridge": "esm_ridge_prediction",
    "Hybrid": "hybrid_prediction",
}


def correlation(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def metrics(y, p):
    i, j = np.triu_indices(len(y), 1)
    dy = y[i] - y[j]
    dp = p[i] - p[j]

    keep = (np.abs(dy) >= 0.03) & (dy != 0)

    pairwise = (
        float(
            np.mean(
                (dy[keep] * dp[keep] > 0)
                + 0.5 * (dp[keep] == 0)
            )
        )
        if keep.any()
        else np.nan
    )

    mse = float(np.mean((y - p) ** 2))

    return {
        "N": len(y),
        "MSE": mse,
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(mse)),
        "Pearson": correlation(y, p),
        "Spearman": correlation(rankdata(y), rankdata(p)),
        "Pairwise@0.03": pairwise,
        "N_pairs": int(keep.sum()),
    }


def read_unique(path, required):
    frame = pd.read_csv(path)

    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    if frame[list(required)].isna().any().any():
        raise ValueError(f"Missing values: {path}")

    if not frame.sequence.is_unique:
        raise ValueError(f"Duplicate sequences: {path}")

    numeric = [c for c in required if c != "sequence"]

    if numeric:
        values = frame[numeric].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values: {path}")

    return frame


def load_predictions(root, split, source, config):
    data = source[["sequence", "target"]].rename(
        columns={"target": "experimental_fitness"}
    ).copy()

    files = [
        ("outputs_rank", "transformer_prediction"),
        ("outputs_esm_ridge", "esm_ridge_prediction"),
        ("outputs_blend", "hybrid_prediction"),
    ]

    for folder, column in files:
        path = root / folder / f"{split}_predictions.csv"

        # The blend script normally saves only validation/test predictions.
        if (
            split == "train"
            and folder == "outputs_blend"
            and not path.exists()
        ):
            continue

        frame = read_unique(
            path,
            {
                "sequence",
                "experimental_fitness",
                "predicted_fitness",
            },
        )

        if set(frame.sequence) != set(data.sequence):
            raise ValueError(
                f"Sequence membership mismatch: {path}"
            )

        frame = frame.set_index("sequence").loc[data.sequence]

        np.testing.assert_allclose(
            data.experimental_fitness.to_numpy(float),
            frame.experimental_fitness.to_numpy(float),
            rtol=0,
            atol=1e-7,
            err_msg=f"Label mismatch: {path}",
        )

        data[column] = frame.predicted_fitness.to_numpy(float)

    expected = (
        config["transformer_weight"]
        * data.transformer_prediction
        + config["esm_weight"]
        * data.esm_ridge_prediction
    )

    if "hybrid_prediction" in data:
        np.testing.assert_allclose(
            data.hybrid_prediction,
            expected,
            rtol=0,
            atol=1e-6,
            err_msg="Saved hybrid predictions disagree with blend weights.",
        )
    else:
        data["hybrid_prediction"] = expected

    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--split-dir", default="official_split")
    parser.add_argument("--out-dir", default="outputs_official_test")
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.bootstraps < 0 or 0 < args.bootstraps < 100:
        parser.error(
            "Use 0 to skip bootstrap, or at least 100 replicates."
        )

    root = args.root.resolve()
    split_dir = root / args.split_dir
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (split_dir / "split_manifest.json").read_text()
    )
    reference = manifest["reference"]

    config = json.loads(
        (root / "outputs_blend/ensemble_config.json").read_text()
    )

    weights = np.array(
        [
            config["transformer_weight"],
            config["esm_weight"],
        ],
        dtype=float,
    )

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError(
            "Blend weights must be finite, nonnegative, and sum to one."
        )

    if len(reference) != 425:
        raise ValueError("Expected a 425-residue reference.")

    # Verify official split sizes and disjoint sequence membership.
    splits = {}

    for split, expected in [
        ("train", 2403),
        ("val", 601),
        ("test", 702),
    ]:
        frame = read_unique(
            split_dir / f"{split}_unseen.csv",
            {"sequence", "target"},
        )

        if len(frame) != expected:
            raise ValueError(
                f"Expected {expected} {split} sequences; got {len(frame)}."
            )

        if not frame.sequence.str.len().eq(len(reference)).all():
            raise ValueError(
                f"Sequence length mismatch in {split}."
            )

        splits[split] = frame

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        if not set(splits[a].sequence).isdisjoint(
            splits[b].sequence
        ):
            raise ValueError(
                f"Sequence overlap between {a} and {b}."
            )

    print("Split sizes and overlap checks passed.", flush=True)

    # Evaluate training, validation, and test predictions.
    rows = []
    matched = {}

    for split, source in splits.items():
        data = load_predictions(root, split, source, config)

        data["mutation_count"] = [
            sum(
                a != b
                for a, b in zip(sequence, reference)
            )
            for sequence in data.sequence
        ]

        def mutation_group(n):
            if n <= 1:
                return "0-1"
            if n == 2:
                return "2"
            if n <= 4:
                return "3-4"
            return "5+"

        data["mutation_group"] = data.mutation_count.map(
            mutation_group
        )

        data.to_csv(
            out / f"{split}_matched_predictions.csv",
            index=False,
        )

        matched[split] = data
        y = data.experimental_fitness.to_numpy(float)

        for model, column in MODELS.items():
            rows.append({
                "Split": split,
                "Model": model,
                **metrics(y, data[column].to_numpy(float)),
            })

    overall = pd.DataFrame(rows)
    overall.to_csv(
        out / "all_split_metrics.csv",
        index=False,
    )

    test_metrics = overall[
        overall.Split.eq("test")
    ].drop(columns="Split")

    test_metrics.to_csv(
        out / "official_test_metrics.csv",
        index=False,
    )

    test = matched["test"]
    test.to_csv(
        out / "official_test_predictions.csv",
        index=False,
    )

    y = test.experimental_fitness.to_numpy(float)

    # Mutation-count subgroups and top-k discovery.
    subgroup_rows = []
    top_rows = []

    high = np.zeros(len(y), dtype=bool)
    n_high = int(np.ceil(0.10 * len(y)))

    high[
        np.argsort(-y, kind="stable")[:n_high]
    ] = True

    for model, column in MODELS.items():
        p = test[column].to_numpy(float)

        for group, subset in test.groupby(
            "mutation_group",
            sort=False,
        ):
            subgroup_rows.append({
                "Model": model,
                "mutation_group": group,
                **metrics(
                    subset.experimental_fitness.to_numpy(float),
                    subset[column].to_numpy(float),
                ),
            })

        order = np.argsort(-p, kind="stable")

        for k in [10, 25, 50]:
            chosen = order[:k]
            precision = float(high[chosen].mean())

            top_rows.append({
                "Model": model,
                "K": k,
                "hits": int(high[chosen].sum()),
                "precision": precision,
                "enrichment": precision / high.mean(),
                "mean_selected_fitness": float(y[chosen].mean()),
            })

    pd.DataFrame(subgroup_rows).to_csv(
        out / "subgroup_metrics.csv",
        index=False,
    )

    pd.DataFrame(top_rows).to_csv(
        out / "official_test_topk.csv",
        index=False,
    )

    # Paired bootstrap: resample the same sequences for every model.
    if args.bootstraps:
        rng = np.random.default_rng(args.seed)

        metric_names = [
            "MAE",
            "MSE",
            "Spearman",
            "Pairwise@0.03",
        ]

        predictions = {
            model: test[column].to_numpy(float)
            for model, column in MODELS.items()
        }

        samples = {
            "Transformer": [],
            "ESM_Ridge": [],
        }

        for iteration in range(args.bootstraps):
            indices = rng.integers(
                0,
                len(y),
                size=len(y),
            )

            scores = {
                model: metrics(
                    y[indices],
                    prediction[indices],
                )
                for model, prediction in predictions.items()
            }

            for baseline in samples:
                samples[baseline].append([
                    scores["Hybrid"][name]
                    - scores[baseline][name]
                    for name in metric_names
                ])

            if (iteration + 1) % 100 == 0:
                print(
                    f"Bootstrap {iteration + 1}/{args.bootstraps}",
                    flush=True,
                )

        intervals = []
        originals = test_metrics.set_index("Model")

        for baseline, draws in samples.items():
            draws = np.asarray(draws)

            for j, name in enumerate(metric_names):
                values = draws[:, j]
                values = values[np.isfinite(values)]

                if len(values) < 0.9 * args.bootstraps:
                    raise ValueError(
                        f"Too few valid bootstrap draws: "
                        f"{baseline}, {name}"
                    )

                intervals.append({
                    "comparison": f"Hybrid - {baseline}",
                    "metric": name,
                    "observed_difference": (
                        originals.loc["Hybrid", name]
                        - originals.loc[baseline, name]
                    ),
                    "ci_lower_95": np.quantile(values, 0.025),
                    "ci_upper_95": np.quantile(values, 0.975),
                    "bootstrap_replicates": len(values),
                })

        pd.DataFrame(intervals).to_csv(
            out / "paired_bootstrap_intervals.csv",
            index=False,
        )

    # Spearman comparison plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(test_metrics.Model, test_metrics.Spearman)
    ax.set(
        ylabel="Spearman correlation",
        title="Official by-mutation test (702 sequences)",
    )
    fig.tight_layout()
    fig.savefig(
        out / "model_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    # Experimental versus predicted fitness plots.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (model, column) in zip(axes, MODELS.items()):
        ax.scatter(
            y,
            test[column],
            s=10,
            alpha=0.5,
        )
        ax.set(
            title=model,
            xlabel="Experimental fitness",
            ylabel="Predicted fitness",
        )

    fig.tight_layout()
    fig.savefig(
        out / "prediction_scatter.png",
        dpi=180,
    )
    plt.close(fig)

    protocol = {
        "dataset": "FLIP2 alpha-amylase",
        "split": "Official by-mutation",
        "split_manifest": manifest,
        "ensemble_config": config,
        "expected_counts": {
            "train": 2403,
            "val": 601,
            "test": 702,
        },
        "evaluation": (
            "Saved predictions; no fitting or model selection "
            "in this evaluator"
        ),
        "prior_official_test_inspection": True,
        "note": (
            "Follow-up experiment after inspection of earlier "
            "official-test results"
        ),
        "pairwise_minimum_fitness_gap": 0.03,
        "topk_high_fitness_definition": (
            "Top ceil(10% of test N); stable ordering for ties"
        ),
        "bootstrap_replicates": args.bootstraps,
        "bootstrap_seed": args.seed,
        "bootstrap_method": (
            "Paired sequence resampling; 95% percentile intervals"
        ),
        "bootstrap_limitation": (
            "Related variants may violate sequence independence"
        ),
    }

    (out / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2)
    )

    print("\nOFFICIAL TEST RESULTS")
    print(
        test_metrics.round(6).to_string(index=False)
    )
    print("\nSaved results to:", out)

    if args.bootstraps:
        print(
            "\nBootstrap differences are Hybrid minus baseline."
            "\nLower MAE/MSE is better."
            "\nHigher Spearman/pairwise accuracy is better."
        )


if __name__ == "__main__":
    main()