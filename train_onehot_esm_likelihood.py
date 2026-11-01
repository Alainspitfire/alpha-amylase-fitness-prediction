"""Train and evaluate a one-hot + ESM-2 likelihood Ridge baseline.

An adapted baseline, not the full published FLIP2 augmented Ridge:
CARP and Dayhoff features are not included.

Run:
    python train_onehot_esm_likelihood.py --root .

Requires the official_split directory and saved test predictions from
outputs_rank, outputs_esm_ridge, and outputs_blend.

Outputs are overwritten when rerun; cached ESM scores are reused.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ALPHABET = "ARNDCQEGHILKMFPSTWYV"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BASELINE_NAME = "OneHot_ESM_Likelihood_Ridge"
EXPECTED_COUNTS = {"train": 2403, "val": 601, "test": 702}


def onehot(sequences):
    lookup = {aa: i for i, aa in enumerate(ALPHABET)}
    sequences = list(sequences)

    columns = np.array([
        [position * 20 + lookup[aa]
         for position, aa in enumerate(sequence)]
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


def load_splits(root):
    parts = {}
    seen = set()

    for split, count in EXPECTED_COUNTS.items():
        path = root / "official_split" / f"{split}_unseen.csv"
        frame = pd.read_csv(path)

        if not {"sequence", "target"}.issubset(frame.columns):
            raise ValueError(f"Missing required columns: {path}")
        if len(frame) != count or not frame.sequence.is_unique:
            raise ValueError(f"Unexpected count or duplicates: {path}")
        if frame.sequence.isna().any():
            raise ValueError(f"Missing sequences: {path}")
        if not all(
            isinstance(s, str)
            and len(s) == 425
            and set(s) <= set(ALPHABET)
            for s in frame.sequence
        ):
            raise ValueError(f"Invalid sequences: {path}")

        frame["target"] = pd.to_numeric(frame.target, errors="raise")
        if not np.isfinite(frame.target.to_numpy(float)).all():
            raise ValueError(f"Non-finite targets: {path}")

        sequences = set(frame.sequence)
        if seen & sequences:
            raise ValueError(f"Sequence overlap involving {split}")
        seen.update(sequences)
        parts[split] = frame

    return parts


def esm_scores(sequences, output, device):
    # Compatible with the cache produced by the original Colab cell.
    cache_path = output / "esm2_unmasked_scores.json"
    scores = (
        json.loads(cache_path.read_text())
        if cache_path.exists()
        else {}
    )

    for sequence in sequences:
        if sequence in scores and not np.isfinite(float(scores[sequence])):
            raise ValueError("Non-finite cached ESM score")

    pending = [s for s in sequences if s not in scores]
    if not pending:
        print("Using cached ESM-2 scores.")
        return scores

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME
    ).to(device).eval()

    for number, sequence in enumerate(
        tqdm(pending, desc="Computing ESM-2 likelihoods"), 1
    ):
        tokens = tokenizer(sequence, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.inference_mode():
            # Unmasked mean log-probability of observed residues.
            # Exclude leading and trailing special tokens.
            logits = model(**tokens).logits[:, 1:-1, :]
            targets = tokens["input_ids"][:, 1:-1]

            if targets.shape[1] != len(sequence):
                raise ValueError("Unexpected tokenizer sequence length")

            score = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            ).item()

        if not np.isfinite(score):
            raise ValueError("Non-finite ESM score")

        scores[sequence] = score

        if number % 50 == 0 or number == len(pending):
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(scores))
            temporary.replace(cache_path)

    del model, tokenizer, tokens, logits, targets
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return scores


def comparison_metrics(actual, predictions):
    i, j = np.triu_indices(len(actual), 1)
    dy = actual[i] - actual[j]
    eligible = np.abs(dy) >= 0.03

    high = np.zeros(len(actual), dtype=bool)
    n_high = int(np.ceil(0.10 * len(actual)))
    high[np.argsort(-actual, kind="stable")[:n_high]] = True

    rows = []
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite predictions: {name}")

        dp = prediction[i] - prediction[j]
        order = np.argsort(-prediction, kind="stable")

        rows.append({
            "Model": name,
            "MSE": np.mean((actual - prediction) ** 2),
            "MAE": np.mean(np.abs(actual - prediction)),
            "Spearman": spearmanr(actual, prediction).statistic,
            "Pairwise@0.03": np.mean(
                (dy[eligible] * dp[eligible] > 0)
                + 0.5 * (dp[eligible] == 0)
            ),
            "Top10_hits": int(high[order[:10]].sum()),
            "Top25_hits": int(high[order[:25]].sum()),
            "Top50_hits": int(high[order[:50]].sum()),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_onehot_esm_likelihood"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.out_dir.expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    parts = load_splits(root)
    actual = parts["test"].target.to_numpy(float)

    # Check existing comparison predictions before expensive scoring.
    other_predictions = {}
    for name, folder in [
        ("Transformer", "outputs_rank"),
        ("ESM_Ridge", "outputs_esm_ridge"),
        ("Hybrid", "outputs_blend"),
    ]:
        path = root / folder / "test_predictions.csv"
        saved = pd.read_csv(path)

        required = {
            "sequence", "experimental_fitness", "predicted_fitness"
        }
        if not required.issubset(saved.columns):
            raise ValueError(f"Missing prediction columns: {path}")
        if (
            not saved.sequence.is_unique
            or set(saved.sequence) != set(parts["test"].sequence)
        ):
            raise ValueError(f"Sequence membership mismatch: {path}")

        saved = saved.set_index("sequence").loc[parts["test"].sequence]
        np.testing.assert_allclose(
            saved.experimental_fitness.to_numpy(float),
            actual,
            rtol=0,
            atol=1e-7,
        )
        prediction = saved.predicted_fitness.to_numpy(float)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite predictions: {path}")
        other_predictions[name] = prediction

    output.mkdir(parents=True, exist_ok=True)
    sequences = [
        sequence
        for frame in parts.values()
        for sequence in frame.sequence
    ]
    scores = esm_scores(sequences, output, device)

    X, y = {}, {}
    for split, frame in parts.items():
        likelihoods = np.array(
            [scores[s] for s in frame.sequence], dtype=float
        ).reshape(-1, 1)

        X[split] = hstack(
            [onehot(frame.sequence), csr_matrix(likelihoods)],
            format="csr",
        )
        y[split] = frame.target.to_numpy(float)

    # Preserve the original experiment's scaling convention.
    # Validation labels contribute to target scaling, but Ridge
    # coefficients are fitted on training rows only.
    # No test labels enter fitting.
    scaler = StandardScaler().fit(
        np.concatenate([y["train"], y["val"]]).reshape(-1, 1)
    )
    z_train = scaler.transform(
        y["train"].reshape(-1, 1)
    ).ravel()

    ridge = Ridge(
        alpha=10.0,
        solver="lsqr",
        tol=1e-5,
        max_iter=1_000_000,
    ).fit(X["train"], z_train)

    baseline_predictions = {}
    for split, frame in parts.items():
        prediction = scaler.inverse_transform(
            ridge.predict(X[split]).reshape(-1, 1)
        ).ravel()
        baseline_predictions[split] = prediction

        pd.DataFrame({
            "sequence": frame.sequence.to_numpy(),
            "experimental_fitness": y[split],
            "predicted_fitness": prediction,
        }).to_csv(output / f"{split}_predictions.csv", index=False)

    predictions = {
        BASELINE_NAME: baseline_predictions["test"],
        **other_predictions,
    }
    results = comparison_metrics(actual, predictions)
    results.to_csv(output / "comparison_metrics.csv", index=False)

    pd.DataFrame({
        "sequence": parts["test"].sequence.to_numpy(),
        "experimental_fitness": actual,
        **predictions,
    }).to_csv(output / "matched_test_predictions.csv", index=False)

    np.savez(
        output / "ridge_head.npz",
        coef=ridge.coef_,
        intercept=ridge.intercept_,
        target_mean=scaler.mean_,
        target_scale=scaler.scale_,
    )

    configuration = {
        "baseline": "one-hot + ESM-2 unmasked mean log-probability",
        "esm_model": MODEL_NAME,
        "alpha": 10.0,
        "solver": "lsqr",
        "tolerance": 1e-5,
        "max_iter": 1_000_000,
        "alphabet": ALPHABET,
        "target_scaling": (
            "training plus validation; Ridge fit on training only"
        ),
        "feature_scaling": "none",
        "full_published_baseline": False,
        "missing_published_features": ["CARP", "combined Dayhoff"],
        "evaluation": "Descriptive comparison; no significance testing",
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2)
    )

    print(results.to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()