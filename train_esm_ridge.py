
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


def encode_embeddings(sequences, model_name, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]

            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False
            )

            tokens = {
                key: value.to(device)
                for key, value in tokens.items()
            }

            residue_mask = tokens["attention_mask"].bool()

            residue_mask &= (
                tokens["input_ids"] != tokenizer.cls_token_id
            )

            residue_mask &= (
                tokens["input_ids"] != tokenizer.eos_token_id
            )

            if tokenizer.pad_token_id is not None:
                residue_mask &= (
                    tokens["input_ids"] != tokenizer.pad_token_id
                )

            output = model(**tokens).last_hidden_state

            weights = residue_mask.unsqueeze(-1)

            pooled = (
                (output * weights).sum(dim=1)
                / weights.sum(dim=1).clamp(min=1)
            )

            embeddings.append(
                pooled.float().cpu().numpy()
            )

            if start == 0 or (start // batch_size + 1) % 50 == 0:
                print(
                    f"Embedded "
                    f"{min(start + batch_size, len(sequences))}"
                    f"/{len(sequences)}",
                    flush=True
                )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--split-dir", default="mutation_holdout")
    parser.add_argument("--out-dir", default="outputs_esm_ridge")

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("batch-size must be positive")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir)

    datasets = {
        name: pd.read_csv(split_dir / f"{name}_unseen.csv")
        for name in ["train", "val", "test"]
    }

    with open(split_dir / "split_manifest.json") as f:
        reference = json.load(f)["reference"]

    sequences = list(dict.fromkeys(
        [reference]
        + [
            seq
            for df in datasets.values()
            for seq in df["sequence"]
        ]
    ))

    cache_key = hashlib.sha256(
        (
            args.model + "\n" + "\n".join(sequences)
        ).encode()
    ).hexdigest()[:20]

    cache_path = output_dir / f"embeddings_{cache_key}.npz"

    if cache_path.exists():
        print("Loading cached embeddings:", cache_path)

        with np.load(cache_path) as cached:
            embeddings = cached["embeddings"]
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("ESM model:", args.model)
        print("Device:", device)

        embeddings = encode_embeddings(
            sequences,
            args.model,
            args.batch_size,
            device
        )

        np.savez_compressed(
            cache_path,
            embeddings=embeddings
        )

    lookup = dict(zip(sequences, embeddings))
    reference_embedding = lookup[reference]

    # Feature = ESM sequence embedding - reference embedding
    # Add mutation count as one additional feature.
    features = {}

    for name, df in datasets.items():
        rows = []

        for sequence in df["sequence"]:
            delta_embedding = (
                lookup[sequence] - reference_embedding
            )

            mutation_count = sum(
                a != b
                for a, b in zip(sequence, reference)
            )

            feature = np.concatenate([
                delta_embedding,
                np.array([mutation_count], dtype=np.float32)
            ])

            rows.append(feature)

        features[name] = np.stack(rows).astype(np.float32)

    # Fit normalization on training features only.
    scaler = StandardScaler()
    scaler.fit(features["train"])

    X = {
        name: scaler.transform(feature)
        for name, feature in features.items()
    }

    y = {
        name: df["target"].to_numpy(dtype=float)
        for name, df in datasets.items()
    }

    # Choose ridge regularization using validation Spearman.
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    best_score = -np.inf
    best_alpha = None

    print("\n===== RIDGE VALIDATION =====")

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X["train"], y["train"])

        val_prediction = model.predict(X["val"])

        score = float(
            spearmanr(
                y["val"],
                val_prediction
            ).statistic
        )

        print(
            f"Alpha: {alpha:8g} | "
            f"Val Spearman: {score:.5f}"
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_alpha = alpha

    if best_alpha is None:
        raise RuntimeError("No valid validation Spearman.")

    print("\nSelected alpha:", best_alpha)

    model = Ridge(alpha=best_alpha)
    model.fit(X["train"], y["train"])

    for name, df in datasets.items():
        predictions = model.predict(X[name])

        pd.DataFrame({
            "sequence": df["sequence"].to_numpy(),
            "experimental_fitness": y[name],
            "predicted_fitness": predictions
        }).to_csv(
            output_dir / f"{name}_predictions.csv",
            index=False
        )

        mse = np.mean((predictions - y[name]) ** 2)
        spearman = spearmanr(
            y[name],
            predictions
        ).statistic

        print(f"\n===== {name.upper()} =====")
        print(f"MSE: {mse:.6f}")
        print(f"Spearman: {spearman:.4f}")

    np.savez(
        output_dir / "ridge_head.npz",
        coef=model.coef_,
        intercept=model.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        alpha=best_alpha
    )

    configuration = {
        "esm_model": args.model,
        "alpha": best_alpha,
        "features": "delta mean embedding + mutation count",
        "selection": "validation Spearman"
    }

    with open(output_dir / "configuration.json", "w") as f:
        json.dump(configuration, f, indent=2)

    print("\nSaved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
