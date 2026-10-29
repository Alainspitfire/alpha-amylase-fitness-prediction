
# predict_hybrid.py

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from data import AA, encode
from model_mutation_effect import MutationEffectTransformer


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridPredictor:
    def __init__(self, root=ROOT):
        self.root = Path(root)

        blend_dir = self.root / "outputs_blend"
        rank_dir = self.root / "outputs_rank"
        esm_dir = self.root / "outputs_esm_ridge"

        with open(blend_dir / "ensemble_config.json") as f:
            self.config = json.load(f)

        self.transformer_weight = float(
            self.config["transformer_weight"]
        )
        self.esm_weight = float(self.config["esm_weight"])

        if not np.isclose(
            self.transformer_weight + self.esm_weight, 1.0
        ):
            raise ValueError("Blending weights must sum to one")

        # Load the three Transformer checkpoints.
        self.transformers = []
        self.reference = None

        for seed in self.config["transformer_seeds"]:
            checkpoint = torch.load(
                rank_dir / f"rank_seed_{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )

            if self.reference is None:
                self.reference = checkpoint["reference"]
            elif checkpoint["reference"] != self.reference:
                raise ValueError("Transformer references disagree")

            model = MutationEffectTransformer(
                reference_tokens=encode(self.reference),
                **checkpoint["model_config"],
            ).to(DEVICE)

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            model.eval()

            self.transformers.append({
                "model": model,
                "mean": float(checkpoint["mean"]),
                "std": float(checkpoint["std"]),
            })

        # Load the fitted Ridge head and feature scaler.
        with np.load(esm_dir / "ridge_head.npz") as ridge:
            self.ridge_coef = ridge["coef"].copy()
            self.ridge_intercept = float(ridge["intercept"])
            self.scaler_mean = ridge["scaler_mean"].copy()
            self.scaler_scale = ridge["scaler_scale"].copy()

        with open(esm_dir / "configuration.json") as f:
            esm_config = json.load(f)

        if esm_config["features"] != (
            "delta mean embedding + mutation count"
        ):
            raise ValueError("Unexpected ESM feature definition")

        self.esm_name = esm_config["esm_model"]

        if self.esm_name != self.config["esm_model"]:
            raise ValueError("ESM backbone mismatch")

        if len(self.reference) != 425:
            raise ValueError("Expected a 425-residue reference")

        if np.any(self.scaler_scale <= 0):
            raise ValueError("Invalid saved feature scaling")

        if (
            self.ridge_coef.size != self.scaler_mean.size
            or self.ridge_coef.size != self.scaler_scale.size
        ):
            raise ValueError("Ridge and scaler dimensions disagree")

        # The pretrained ESM backbone is loaded separately.
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm_name)
        self.esm = AutoModel.from_pretrained(
            self.esm_name
        ).to(DEVICE).eval()

        self.reference_embedding = self._embed(
            [self.reference]
        )[0]

        if self.ridge_coef.size != self.reference_embedding.size + 1:
            raise ValueError("ESM embedding and Ridge dimensions disagree")

    def _validate(self, sequence):
        sequence = sequence.strip().upper()

        if len(sequence) != len(self.reference):
            raise ValueError(
                f"Expected {len(self.reference)} residues, "
                f"received {len(sequence)}"
            )

        invalid = set(sequence) - set(AA)
        if invalid:
            raise ValueError(
                f"Unsupported amino acids: {sorted(invalid)}"
            )

        return sequence

    @torch.inference_mode()
    def _embed(self, sequences):
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        residue_mask = tokens["attention_mask"].bool()

        for special_id in (
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ):
            if special_id is not None:
                residue_mask &= tokens["input_ids"] != special_id

        hidden = self.esm(**tokens).last_hidden_state
        mask = residue_mask.unsqueeze(-1)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return pooled.float().cpu().numpy()

    @torch.inference_mode()
    def predict(self, sequence):
        sequence = self._validate(sequence)

        x = torch.tensor(
            [encode(sequence)],
            dtype=torch.long,
            device=DEVICE,
        )

        # Transformer ensemble.
        transformer_predictions = []

        for entry in self.transformers:
            prediction_z = entry["model"](x).item()

            prediction = (
                prediction_z * entry["std"] + entry["mean"]
            )
            transformer_predictions.append(prediction)

        transformer_fitness = float(
            np.mean(transformer_predictions)
        )

        # ESM-2 embedding difference from reference.
        sequence_embedding = self._embed([sequence])[0]
        delta = sequence_embedding - self.reference_embedding

        mutation_count = sum(
            a != b
            for a, b in zip(sequence, self.reference)
        )

        features = np.concatenate([
            delta,
            np.array([mutation_count], dtype=np.float32),
        ])

        # Apply training-fitted StandardScaler.
        scaled = (
            features - self.scaler_mean
        ) / self.scaler_scale

        esm_fitness = float(
            np.dot(scaled, self.ridge_coef)
            + self.ridge_intercept
        )

        # Apply validation-selected ensemble weights.
        hybrid_fitness = (
            self.transformer_weight * transformer_fitness
            + self.esm_weight * esm_fitness
        )

        return {
            "predicted_fitness": float(hybrid_fitness),
            "transformer_fitness": transformer_fitness,
            "esm_ridge_fitness": esm_fitness,
            "transformer_weight": self.transformer_weight,
            "esm_weight": self.esm_weight,
            "mutation_count": mutation_count,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        required=True,
        help="425-residue alpha-amylase amino-acid sequence",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Directory containing trained model outputs",
    )
    args = parser.parse_args()

    predictor = HybridPredictor(args.root)
    result = predictor.predict(args.sequence)

    print("\n===== HYBRID ALPHA-AMYLASE PREDICTION =====")
    print(f"Predicted fitness: {result['predicted_fitness']:.6f}")
    print(
        f"Transformer prediction: "
        f"{result['transformer_fitness']:.6f}"
    )
    print(
        f"ESM Ridge prediction: "
        f"{result['esm_ridge_fitness']:.6f}"
    )
    print(
        f"Blend weights: "
        f"{result['transformer_weight']:.2f} / "
        f"{result['esm_weight']:.2f}"
    )
    print(f"Mutations vs reference: {result['mutation_count']}")


if __name__ == "__main__":
    main()


def predict_mutation_effect(sequence_a, sequence_b, predictor):
    """
    Predict the change in fitness from protein A to protein B.

    Positive delta: B is predicted to have higher fitness.
    Negative delta: B is predicted to have lower fitness.
    """

    sequence_a = sequence_a.strip().upper()
    sequence_b = sequence_b.strip().upper()

    if len(sequence_a) != len(sequence_b):
        raise ValueError("Sequences must have equal length.")

    result_a = predictor.predict(sequence_a)
    result_b = predictor.predict(sequence_b)

    fitness_a = float(result_a["predicted_fitness"])
    fitness_b = float(result_b["predicted_fitness"])

    delta = fitness_b - fitness_a

    mutations = [
        f"{aa_a}{i}{aa_b}"
        for i, (aa_a, aa_b) in enumerate(
            zip(sequence_a, sequence_b), start=1
        )
        if aa_a != aa_b
    ]

    # Screening threshold, not a statistical significance test.
    threshold = 0.03

    if delta >= threshold:
        category = "Predicted substantial increase"
    elif delta <= -threshold:
        category = "Predicted substantial decrease"
    else:
        category = "Predicted small change"

    return {
        "fitness_a": fitness_a,
        "fitness_b": fitness_b,
        "delta_fitness": delta,
        "mutation_count_between_sequences": len(mutations),
        "mutations": mutations,
        "category": category,
    }
