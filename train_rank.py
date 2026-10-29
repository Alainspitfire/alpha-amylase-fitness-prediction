
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader

from data import ProteinDataset, encode
from model_mutation_effect import MutationEffectTransformer


SEEDS = [42, 123, 2026]

CONFIG = {
    "vocab_size": 22,
    "max_len": 425,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1
}


def pairwise_rank_loss(pred, target, min_gap=0.0):
    true_diff = target[:, None] - target[None, :]
    pred_diff = pred[:, None] - pred[None, :]

    mask = torch.triu(
        torch.ones_like(true_diff, dtype=torch.bool),
        diagonal=1
    ) & (true_diff.abs() > min_gap)

    if not bool(mask.any()):
        return pred.sum() * 0.0

    signs = torch.sign(true_diff[mask])

    return F.softplus(
        -signs * pred_diff[mask]
    ).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict(model, loader, device, mean, std):
    model.eval()
    predictions = []

    for x, mask, y in loader:
        x = x.to(device)
        mask = mask.to(device)

        output = model(x, mask)
        output = output * std + mean

        predictions.append(output.cpu().numpy())

    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split-dir", default="mutation_holdout")
    parser.add_argument("--out-dir", default="outputs_rank")
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    if args.rank_weight < 0 or args.min_gap < 0:
        parser.error("rank-weight and min-gap must be nonnegative")

    split_dir = Path(args.split_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(split_dir / "train_unseen.csv")
    val_df = pd.read_csv(split_dir / "val_unseen.csv")
    test_df = pd.read_csv(split_dir / "test_unseen.csv")

    with open(split_dir / "split_manifest.json") as f:
        reference = json.load(f)["reference"]

    mean = float(train_df["target"].mean())
    std = float(train_df["target"].std())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    datasets = {
        "train": ProteinDataset(train_df, reference, mean, std),
        "val": ProteinDataset(val_df, reference, mean, std),
        "test": ProteinDataset(test_df, reference, mean, std)
    }

    fixed_loaders = {
        name: DataLoader(dataset, batch_size=32, shuffle=False)
        for name, dataset in datasets.items()
    }

    all_predictions = {
        "train": [],
        "val": [],
        "test": []
    }

    print("Device:", device)
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    for seed in SEEDS:
        set_seed(seed)

        model = MutationEffectTransformer(
            reference_tokens=encode(reference),
            **CONFIG
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4
        )

        train_loader = DataLoader(
            datasets["train"],
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed)
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        print(f"\n===== TRAINING SEED {seed} =====")

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for x, mask, y in train_loader:
                x = x.to(device)
                mask = mask.to(device)
                y = y.to(device)

                prediction = model(x, mask)

                mse = F.mse_loss(prediction, y)

                ranking = pairwise_rank_loss(
                    prediction,
                    y,
                    min_gap=args.min_gap / std
                )

                loss = mse + args.rank_weight * ranking

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)

            model.eval()
            val_loss_sum = 0.0

            with torch.inference_mode():
                for x, mask, y in fixed_loaders["val"]:
                    prediction = model(
                        x.to(device),
                        mask.to(device)
                    )

                    val_loss_sum += F.mse_loss(
                        prediction,
                        y.to(device),
                        reduction="sum"
                    ).item()

            val_loss = val_loss_sum / len(datasets["val"])

            print(
                f"Epoch {epoch:02d} | "
                f"Train loss: {total_loss / len(datasets['train']):.5f} | "
                f"Val MSE: {val_loss:.5f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print("Early stopping.")
                break

        model.load_state_dict(best_state)

        torch.save(
            {
                "model_state_dict": best_state,
                "model_config": CONFIG,
                "reference": reference,
                "mean": mean,
                "std": std,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "rank_weight": args.rank_weight,
                "min_gap": args.min_gap
            },
            output_dir / f"rank_seed_{seed}.pt"
        )

        for name, loader in fixed_loaders.items():
            all_predictions[name].append(
                predict(model, loader, device, mean, std)
            )

        print(
            f"Best epoch: {best_epoch}, "
            f"Val MSE: {best_val_loss:.5f}"
        )

    for name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df)
    ]:
        predictions = np.mean(
            np.stack(all_predictions[name]),
            axis=0
        )

        pd.DataFrame({
            "sequence": df["sequence"].to_numpy(),
            "experimental_fitness": df["target"].to_numpy(),
            "predicted_fitness": predictions
        }).to_csv(
            output_dir / f"{name}_predictions.csv",
            index=False
        )

        y = df["target"].to_numpy(dtype=float)

        mse = np.mean((y - predictions) ** 2)
        mae = np.mean(np.abs(y - predictions))

        print(f"\n===== {name.upper()} =====")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Pearson: {pearsonr(y, predictions).statistic:.4f}")
        print(f"Spearman: {spearmanr(y, predictions).statistic:.4f}")

    print("\nSaved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
