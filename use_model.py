
"""Alpha-amylase fitness inference CLI: Transformer-only or hybrid ensemble."""

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = (42, 123, 2026)

TRANSFORMER_FILES = tuple(
    f"outputs_rank/rank_seed_{s}.pt" for s in SEEDS
)

HYBRID_FILES = (
    "outputs_esm_ridge/ridge_head.npz",
    "outputs_esm_ridge/configuration.json",
    "outputs_blend/ensemble_config.json",
)


# ============================================================
# 1. REQUIRED FILES AND INSTALLATION
# ============================================================

def requirements(model):
    source = ["data.py", "model_mutation_effect.py"]
    weights = list(TRANSFORMER_FILES)

    if model == "hybrid":
        source.append("predict_hybrid.py")
        weights.extend(HYBRID_FILES)

    return source, weights


def install_weights(archive_path, root):
    """Unpack recognized weights from a trained-model ZIP."""

    all_targets = TRANSFORMER_FILES + HYBRID_FILES

    with zipfile.ZipFile(archive_path) as zf:
        entries = {}

        for info in zf.infolist():
            if info.is_dir():
                continue

            for target in all_targets:
                parts = Path(target).parts

                if Path(info.filename).parts[-len(parts):] == parts:
                    if target in entries:
                        raise ValueError(f"Duplicate ZIP entry: {target}")

                    entries[target] = info

        if not all(
            target in entries for target in TRANSFORMER_FILES
        ):
            raise ValueError(
                "ZIP is missing Transformer checkpoint files"
            )

        for target, info in entries.items():
            destination = root / target
            destination.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info) as source:
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

            print(f"Installed {target}")


def check_files(root, model):
    sources, weights = requirements(model)

    missing = [
        f for f in sources + weights
        if not (root / f).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files for " + model + ":\n  "
            + "\n  ".join(missing)
            + "\nDownload the repository source and "
              "model weights, then run setup."
        )


# ============================================================
# 2. LET USER CHOOSE MODEL
# ============================================================

def select_model(value):
    if value:
        return value

    if not sys.stdin.isatty():
        raise ValueError(
            "Specify --model transformer or --model hybrid "
            "when not interactive"
        )

    print("\n===== SELECT PREDICTION MODEL =====")
    print("1. Mutation-aware Transformer")
    print("   Faster; no ESM-2 required")
    print()
    print("2. Hybrid Transformer + ESM-2 Ridge")
    print("   Better overall fitness ranking")

    while True:
        choice = input("\nEnter 1 or 2: ").strip().lower()

        if choice in ("1", "transformer"):
            return "transformer"

        if choice in ("2", "hybrid"):
            return "hybrid"

        print("Please enter 1 or 2.")


# ============================================================
# 3. LOAD STANDALONE TRANSFORMER ENSEMBLE
# ============================================================

class TransformerOnly:
    """Load the three ranking-trained Transformer checkpoints."""

    def __init__(self, root):
        import torch

        from data import encode
        from model_mutation_effect import MutationEffectTransformer

        self.torch = torch

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.models = []
        self.stats = []
        self.reference = None

        for relative in TRANSFORMER_FILES:
            path = root / relative

            # Only use trusted checkpoint files.
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            reference = checkpoint["reference"]

            if self.reference is None:
                self.reference = reference

            elif reference != self.reference:
                raise ValueError(
                    "Reference sequences differ between checkpoints"
                )

            model = MutationEffectTransformer(
                reference_tokens=encode(reference),
                **checkpoint["model_config"],
            )

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            model.to(self.device)
            model.eval()

            self.models.append(model)

            self.stats.append((
                float(checkpoint["mean"]),
                float(checkpoint["std"]),
            ))

    def predict(self, sequence):
        from data import encode

        torch = self.torch

        x = torch.tensor(
            [encode(sequence)],
            dtype=torch.long,
            device=self.device,
        )

        values = []

        with torch.inference_mode():
            for model, (mean, std) in zip(
                self.models, self.stats
            ):
                normalized = model(x)

                fitness = (
                    normalized.item() * std + mean
                )

                values.append(float(fitness))

        return {
            "predicted_fitness": sum(values) / len(values)
        }


# ============================================================
# 4. LOAD SELECTED MODEL
# ============================================================

def load_model(root, model):
    check_files(root, model)

    sys.path.insert(0, str(root))

    if model == "transformer":
        return TransformerOnly(root)

    from predict_hybrid import HybridPredictor

    return HybridPredictor(root)


# ============================================================
# 5. READ AND VALIDATE PROTEIN SEQUENCES
# ============================================================

def read_sequence(sequence, fasta):
    if sequence and fasta:
        raise ValueError(
            "Provide --sequence or --fasta, not both"
        )

    if fasta:
        lines = Path(fasta).read_text().splitlines()

        records = []
        current = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current:
                    records.append("".join(current))
                    current = []
            else:
                current.append(line)

        if current:
            records.append("".join(current))

        if len(records) != 1:
            raise ValueError(
                "FASTA must contain exactly one protein"
            )

        sequence = records[0]

    if sequence:
        return "".join(sequence.split()).upper()

    return None


def validate(sequence, reference):
    if len(sequence) != len(reference):
        raise ValueError(
            f"Expected length {len(reference)}, "
            f"received {len(sequence)}"
        )

    invalid = sorted(set(sequence) - set(AA))

    if invalid:
        raise ValueError(
            f"Invalid amino-acid symbols: {invalid}"
        )


# ============================================================
# 6. FITNESS PREDICTION
# ============================================================

def fitness(predictor, sequence, model):
    validate(sequence, predictor.reference)

    raw = predictor.predict(sequence)

    result = {
        "model": model,
        "predicted_fitness": float(
            raw["predicted_fitness"]
        ),
    }

    if model == "hybrid":
        result["transformer_fitness"] = float(
            raw["transformer_fitness"]
        )

        result["esm_ridge_fitness"] = float(
            raw["esm_ridge_fitness"]
        )

    return result


# ============================================================
# 7. COMMAND-LINE INTERFACE
# ============================================================

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory holding model scripts and checkpoints",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # SETUP
    setup = sub.add_parser(
        "setup",
        help="Install/check required model files",
    )

    setup.add_argument(
        "--weights-zip",
        type=Path,
    )

    setup.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
        default="hybrid",
    )

    # FITNESS
    pfit = sub.add_parser(
        "fitness",
        help="Predict fitness for one sequence",
    )

    pfit.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
    )

    pfit.add_argument("--sequence")
    pfit.add_argument("--fasta", type=Path)

    # COMPARE TWO MUTANTS
    pcmp = sub.add_parser(
        "compare",
        help="Compare mutant B versus mutant A",
    )

    pcmp.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
    )

    pcmp.add_argument(
        "--sequence-a",
        required=True,
    )

    pcmp.add_argument(
        "--sequence-b",
        required=True,
    )

    # SCREEN MUTATIONS
    pscreen = sub.add_parser(
        "screen",
        help="Rank all 19 substitutions at one position",
    )

    pscreen.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
    )

    pscreen.add_argument(
        "--position",
        type=int,
        required=True,
    )

    pscreen.add_argument(
        "--output",
        type=Path,
        default=Path("screening_results.csv"),
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    if args.command == "setup":
        if args.weights_zip:
            install_weights(args.weights_zip, root)

        check_files(root, args.model)

        print(f"Setup complete for {args.model}.")

        print(
            "Install dependencies: "
            "python -m pip install -r requirements.txt"
        )

        print(
            "Hybrid inference may download ESM-2 "
            "from Hugging Face on first use."
        )

        return

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    model = select_model(args.model)

    print(f"\nSelected model: {model}")

    predictor = load_model(root, model)

    reference = predictor.reference

    # --------------------------------------------------------
    # FITNESS
    # --------------------------------------------------------

    if args.command == "fitness":
        sequence = (
            read_sequence(args.sequence, args.fasta)
            or reference
        )

        result = fitness(
            predictor,
            sequence,
            model,
        )

        print(json.dumps(result, indent=2))

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    elif args.command == "compare":
        a = read_sequence(args.sequence_a, None)
        b = read_sequence(args.sequence_b, None)

        result_a = fitness(predictor, a, model)
        result_b = fitness(predictor, b, model)

        changes = [
            f"{x}{i}{y}"
            for i, (x, y) in enumerate(zip(a, b), 1)
            if x != y
        ]

        delta = (
            result_b["predicted_fitness"]
            - result_a["predicted_fitness"]
        )

        print(json.dumps({
            "model": model,
            "fitness_a": result_a["predicted_fitness"],
            "fitness_b": result_b["predicted_fitness"],
            "delta_b_minus_a": delta,
            "differences": changes,
            "note": (
                "Predicted difference, not an "
                "experimentally confirmed mutation effect."
            ),
        }, indent=2))

    # --------------------------------------------------------
    # SCREEN
    # --------------------------------------------------------

    else:
        position = args.position

        if not 1 <= position <= len(reference):
            raise ValueError(
                f"Position must be between 1 and {len(reference)}"
            )

        original = reference[position - 1]

        ref_score = fitness(
            predictor,
            reference,
            model,
        )["predicted_fitness"]

        rows = []

        for aa in AA:
            if aa == original:
                continue

            mutated = (
                reference[:position - 1]
                + aa
                + reference[position:]
            )

            score = fitness(
                predictor,
                mutated,
                model,
            )["predicted_fitness"]

            rows.append({
                "mutation": f"{original}{position}{aa}",
                "predicted_fitness": score,
                "predicted_delta_vs_reference": (
                    score - ref_score
                ),
            })

        rows.sort(
            key=lambda row: row["predicted_fitness"],
            reverse=True,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "model",
                    "mutation",
                    "predicted_fitness",
                    "predicted_delta_vs_reference",
                ],
            )

            writer.writeheader()

            for rank, row in enumerate(rows, 1):
                writer.writerow({
                    "rank": rank,
                    "model": model,
                    **row,
                })

        print(
            f"Saved {len(rows)} ranked substitutions "
            f"using {model}: {args.output}"
        )

        for rank, row in enumerate(rows[:5], 1):
            print(
                f"{rank}. {row['mutation']} | "
                f"fitness={row['predicted_fitness']:.6f} | "
                f"delta={row['predicted_delta_vs_reference']:+.6f}"
            )


if __name__ == "__main__":
    cli()
