
# blend_predictions.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_aligned(first_path, second_path):
    a = pd.read_csv(first_path)
    b = pd.read_csv(second_path)

    required = {
        "sequence",
        "experimental_fitness",
        "predicted_fitness",
    }

    for name, df in [("Transformer", a), ("ESM", b)]:
        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns in {name}: "
                f"{required - set(df.columns)}"
            )
        if df["sequence"].duplicated().any():
            raise ValueError(f"Duplicate sequences in {name}")

    merged = a.merge(
        b,
        on="sequence",
        suffixes=("_rank", "_esm"),
        validate="one_to_one",
    )

    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Prediction files contain different sequences")

    y = merged["experimental_fitness_rank"].to_numpy(float)

    if not np.allclose(
        y,
        merged["experimental_fitness_esm"].to_numpy(float),
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError("Experimental fitness values disagree")

    return (
        merged["sequence"].to_numpy(),
        y,
        merged["predicted_fitness_rank"].to_numpy(float),
        merged["predicted_fitness_esm"].to_numpy(float),
    )


def pairwise_accuracy(y, prediction, delta=0.03):
    i, j = np.triu_indices(len(y), k=1)

    true_diff = y[i] - y[j]
    pred_diff = prediction[i] - prediction[j]

    valid = np.abs(true_diff) >= delta

    if not valid.any():
        return float("nan")

    correct = (true_diff[valid] * pred_diff[valid] > 0).sum()
    ties = (pred_diff[valid] == 0).sum()

    return float((correct + 0.5 * ties) / valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-dir", default="outputs_rank")
    parser.add_argument("--esm-dir", default="outputs_esm_ridge")
    parser.add_argument("--out-dir", default="outputs_blend")

    args = parser.parse_args()

    rank_dir = Path(args.rank_dir)
    esm_dir = Path(args.esm_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {}

    for split in ("val", "test"):
        data[split] = load_aligned(
            rank_dir / f"{split}_predictions.csv",
            esm_dir / f"{split}_predictions.csv",
        )

    _, y_val, rank_val, esm_val = data["val"]

    best_weight = None
    best_score = -np.inf

    print("===== VALIDATION BLENDING =====")

    for weight in np.linspace(0.0, 1.0, 11):
        prediction = weight * rank_val + (1 - weight) * esm_val
        score = float(spearmanr(y_val, prediction).statistic)

        print(
            f"Transformer weight: {weight:.1f} | "
            f"Validation Spearman: {score:.5f}"
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_weight = float(weight)

    if best_weight is None:
        raise RuntimeError("No valid validation Spearman")

    config = {
        "model_type": "Transformer_ESM2_Ridge_Hybrid",
        "transformer_weight": best_weight,
        "esm_weight": 1.0 - best_weight,
        "transformer_seeds": [42, 123, 2026],
        "esm_model": "facebook/esm2_t33_650M_UR50D",
        "validation_spearman": best_score,
        "selection_metric": "validation_spearman",
        "feature_definition": "delta mean embedding + mutation count",
    }

    with open(output_dir / "ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n===== SELECTED ENSEMBLE =====")
    print("Transformer weight:", best_weight)
    print("ESM weight:", 1.0 - best_weight)
    print("Validation Spearman:", best_score)

    for split, (sequences, y, rank_pred, esm_pred) in data.items():
        prediction = (
            best_weight * rank_pred
            + (1.0 - best_weight) * esm_pred
        )

        mse = float(np.mean((y - prediction) ** 2))
        mae = float(np.mean(np.abs(y - prediction)))
        spearman = float(spearmanr(y, prediction).statistic)
        pairwise = pairwise_accuracy(y, prediction)

        print(f"\n===== {split.upper()} =====")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Spearman: {spearman:.5f}")
        print(f"Pairwise accuracy: {pairwise:.2%}")

        pd.DataFrame({
            "sequence": sequences,
            "experimental_fitness": y,
            "predicted_fitness": prediction,
        }).to_csv(
            output_dir / f"{split}_predictions.csv",
            index=False,
        )

    print("\nSaved ensemble configuration and predictions:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
