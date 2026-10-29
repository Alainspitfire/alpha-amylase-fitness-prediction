"""Alpha-amylase fitness prediction, comparison, screening, and ranking."""

import argparse
import csv
import json
import math
import random
import shutil
import sys
import zipfile
from pathlib import Path


AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = (42, 123, 2026)

TRANSFORMER_FILES = tuple(
    f"outputs_rank/rank_seed_{seed}.pt"
    for seed in SEEDS
)

HYBRID_FILES = (
    "outputs_esm_ridge/ridge_head.npz",
    "outputs_esm_ridge/configuration.json",
    "outputs_blend/ensemble_config.json",
)


def requirements(model):
    sources = ["data.py", "model_mutation_effect.py"]
    weights = list(TRANSFORMER_FILES)

    if model == "hybrid":
        sources.append("predict_hybrid.py")
        weights.extend(HYBRID_FILES)

    return sources, weights


def install_weights(archive_path, root):
    """Install recognized model files from a ZIP archive."""
    targets = TRANSFORMER_FILES + HYBRID_FILES

    with zipfile.ZipFile(archive_path) as archive:
        entries = {}

        for info in archive.infolist():
            if info.is_dir():
                continue

            for target in targets:
                parts = Path(target).parts

                if Path(info.filename).parts[-len(parts):] == parts:
                    if target in entries:
                        raise ValueError(
                            f"Duplicate ZIP entry: {target}"
                        )

                    entries[target] = info

        if not all(
            target in entries for target in TRANSFORMER_FILES
        ):
            raise ValueError(
                "ZIP is missing Transformer checkpoint files"
            )

        for target, info in entries.items():
            destination = root / target
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(info) as source:
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

            print(f"Installed {target}")


def check_files(root, model):
    sources, weights = requirements(model)

    missing = [
        name
        for name in sources + weights
        if not (root / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing files for {model}:\n  "
            + "\n  ".join(missing)
        )


def select_model(value):
    if value:
        return value

    if not sys.stdin.isatty():
        raise ValueError(
            "Specify --model transformer or --model hybrid "
            "when not interactive"
        )

    print("\nSelect prediction model:")
    print("1. Mutation-aware Transformer ensemble")
    print("2. Hybrid Transformer + ESM–Ridge")

    while True:
        choice = input("Enter 1 or 2: ").strip().lower()

        if choice in ("1", "transformer"):
            return "transformer"

        if choice in ("2", "hybrid"):
            return "hybrid"

        print("Please enter 1 or 2.")


class TransformerOnly:
    """Load the three Transformer checkpoints once."""

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
            checkpoint = torch.load(
                root / relative,
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
                self.models,
                self.stats,
            ):
                normalized = model(x)
                value = normalized.item() * std + mean
                values.append(float(value))

        return {
            "predicted_fitness": sum(values) / len(values)
        }


def load_model(root, model):
    check_files(root, model)
    sys.path.insert(0, str(root))

    if model == "transformer":
        return TransformerOnly(root)

    from predict_hybrid import HybridPredictor

    return HybridPredictor(root)


def read_sequence(sequence, fasta):
    if sequence and fasta:
        raise ValueError(
            "Provide --sequence or --fasta, not both"
        )

    if fasta:
        records = []
        current = []

        for line in Path(fasta).read_text().splitlines():
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


def read_ranking_input(
    csv_path=None,
    fasta_path=None,
    sequence_column="sequence",
    fitness_column=None,
):
    """Read CSV or multi-record FASTA candidates."""
    rows = []

    if fasta_path:
        identifier = None
        fragments = []

        def append_record():
            if identifier is None:
                return

            if not fragments:
                raise ValueError(
                    f"Empty FASTA record: {identifier}"
                )

            rows.append({
                "id": identifier,
                "source_row": len(rows) + 1,
                "sequence": "".join(fragments).upper(),
            })

        for line in Path(fasta_path).read_text().splitlines():
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                append_record()

                identifier = (
                    line[1:].strip()
                    or f"sequence_{len(rows) + 1}"
                )
                fragments = []
            else:
                if identifier is None:
                    raise ValueError(
                        "FASTA sequence appears before its header"
                    )

                fragments.append("".join(line.split()))

        append_record()

    else:
        with Path(csv_path).open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []

            if sequence_column not in fields:
                raise ValueError(
                    "CSV is missing sequence column: "
                    f"{sequence_column}"
                )

            if fitness_column is None:
                fitness_column = next(
                    (
                        name
                        for name in (
                            "experimental_fitness",
                            "target",
                        )
                        if name in fields
                    ),
                    None,
                )
            elif fitness_column not in fields:
                raise ValueError(
                    "CSV is missing fitness column: "
                    f"{fitness_column}"
                )

            for number, record in enumerate(reader, 1):
                sequence = "".join(
                    (record.get(sequence_column) or "").split()
                ).upper()

                row = {
                    "id": (
                        record.get("id")
                        or f"sequence_{number}"
                    ),
                    "source_row": number,
                    "sequence": sequence,
                }

                if fitness_column is not None:
                    try:
                        actual = float(
                            record.get(fitness_column, "")
                        )
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Invalid experimental fitness "
                            f"in CSV row {number}"
                        ) from error

                    if not math.isfinite(actual):
                        raise ValueError(
                            "Non-finite experimental fitness "
                            f"in CSV row {number}"
                        )

                    row["experimental_fitness"] = actual

                rows.append(row)

    if not rows:
        raise ValueError("No candidate sequences found")

    return rows


def descending_ranks(values):
    """Higher scores rank first. Ties share ranks: 1, 1, 3."""
    ranks = {}

    for number, value in enumerate(
        sorted(values, reverse=True),
        1,
    ):
        ranks.setdefault(value, number)

    return ranks


def rank_sequences(predictor, rows, model, output):
    """Predict each unique sequence once and rank candidates."""

    # Validate all selected sequences before starting predictions.
    for row in rows:
        try:
            validate(
                row["sequence"],
                predictor.reference,
            )
        except ValueError as error:
            raise ValueError(
                f"Candidate {row['id']} "
                f"(row {row['source_row']}): {error}"
            ) from error

    cache = {}

    for number, row in enumerate(rows, 1):
        sequence = row["sequence"]

        if sequence not in cache:
            result = fitness(
                predictor,
                sequence,
                model,
            )

            if not all(
                math.isfinite(value)
                for key, value in result.items()
                if key != "model"
            ):
                raise ValueError(
                    "Non-finite prediction for "
                    f"candidate {row['id']}"
                )

            cache[sequence] = result

        row.update(cache[sequence])

        row["mutation_count"] = sum(
            a != b
            for a, b in zip(
                sequence,
                predictor.reference,
            )
        )

        if (
            number == 1
            or number % 10 == 0
            or number == len(rows)
        ):
            print(
                f"Predicted {number}/{len(rows)} candidates",
                flush=True,
            )

    prediction_ranks = descending_ranks([
        row["predicted_fitness"]
        for row in rows
    ])

    has_labels = all(
        "experimental_fitness" in row
        for row in rows
    )

    if has_labels:
        actual_ranks = descending_ranks([
            row["experimental_fitness"]
            for row in rows
        ])

    for row in rows:
        row["predicted_rank"] = prediction_ranks[
            row["predicted_fitness"]
        ]

        if has_labels:
            row["experimental_rank"] = actual_ranks[
                row["experimental_fitness"]
            ]

            row["absolute_error"] = abs(
                row["predicted_fitness"]
                - row["experimental_fitness"]
            )

    # Stable ordering preserves input order when scores tie.
    rows.sort(
        key=lambda row: row["predicted_fitness"],
        reverse=True,
    )

    fields = [
        "predicted_rank",
        "model",
        "id",
        "source_row",
        "sequence",
        "predicted_fitness",
    ]

    if model == "hybrid":
        fields += [
            "transformer_fitness",
            "esm_ridge_fitness",
        ]

    fields += ["mutation_count"]

    if has_labels:
        fields += [
            "experimental_fitness",
            "experimental_rank",
            "absolute_error",
        ]

    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved {len(rows)} candidates "
        f"({len(cache)} unique sequences): {output}"
    )

    print(
        "Ranks are within these candidates only; "
        "higher fitness ranks first."
    )

    print(
        "Ties share ranks (1, 1, 3); "
        "tied rows retain their input order."
    )

    for row in rows[:10]:
        actual = (
            f" | experimental={row['experimental_fitness']:.6f}"
            if has_labels
            else ""
        )

        print(
            f"{row['predicted_rank']}. {row['id']} | "
            f"fitness={row['predicted_fitness']:.6f}"
            f"{actual}"
        )

    return rows


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing model scripts and checkpoints",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Setup.
    setup = sub.add_parser(
        "setup",
        help="Install/check model files",
    )
    setup.add_argument("--weights-zip", type=Path)
    setup.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
        default="hybrid",
    )

    # Predict one sequence.
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

    # Compare two sequences.
    pcmp = sub.add_parser(
        "compare",
        help="Compare mutant B against mutant A",
    )
    pcmp.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
    )
    pcmp.add_argument("--sequence-a", required=True)
    pcmp.add_argument("--sequence-b", required=True)

    # Screen substitutions at one reference position.
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

    # Predict and rank a batch of existing sequences.
    prank = sub.add_parser(
        "rank",
        help="Predict and rank CSV or multi-record FASTA candidates",
    )
    prank.add_argument(
        "--model",
        choices=("transformer", "hybrid"),
        default="hybrid",
    )

    inputs = prank.add_mutually_exclusive_group()

    inputs.add_argument(
        "--csv",
        type=Path,
        help="Default: ROOT/official_split/test_unseen.csv",
    )
    inputs.add_argument(
        "--fasta",
        type=Path,
        help="Multi-record FASTA containing candidate sequences",
    )

    prank.add_argument(
        "--sequence-column",
        default="sequence",
    )
    prank.add_argument(
        "--fitness-column",
        help=(
            "Optional experimental label column; "
            "auto-detect experimental_fitness or target"
        ),
    )
    prank.add_argument(
        "--sample",
        type=int,
        help="Randomly select this many candidates before inference",
    )
    prank.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    prank.add_argument(
        "--output",
        type=Path,
        default=Path("ranked_predictions.csv"),
    )

    args = parser.parse_args()
    root = args.root.expanduser().resolve()

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

    ranking_rows = None

    if args.command == "rank":
        if args.sample is not None and args.sample < 1:
            parser.error("--sample must be positive")

        if args.fasta and args.fitness_column:
            parser.error(
                "--fitness-column applies only to CSV input"
            )

        csv_path = args.csv or (
            None
            if args.fasta
            else root / "official_split/test_unseen.csv"
        )

        input_path = Path(
            args.fasta or csv_path
        ).expanduser().resolve()

        if args.output.expanduser().resolve() == input_path:
            parser.error(
                "Output must differ from the input file"
            )

        ranking_rows = read_ranking_input(
            csv_path=(
                input_path if not args.fasta else None
            ),
            fasta_path=(
                input_path if args.fasta else None
            ),
            sequence_column=args.sequence_column,
            fitness_column=args.fitness_column,
        )

        if args.sample is not None:
            if args.sample > len(ranking_rows):
                parser.error(
                    "--sample exceeds input size "
                    f"({len(ranking_rows)})"
                )

            indices = sorted(
                random.Random(args.seed).sample(
                    range(len(ranking_rows)),
                    args.sample,
                )
            )

            ranking_rows = [
                ranking_rows[index]
                for index in indices
            ]

    model = select_model(args.model)

    print(f"\nSelected model: {model}")

    # Load once for the entire command.
    predictor = load_model(root, model)
    reference = predictor.reference

    if args.command == "rank":
        rank_sequences(
            predictor,
            ranking_rows,
            model,
            args.output.expanduser(),
        )

    elif args.command == "fitness":
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
                "Predicted difference, not an experimentally "
                "confirmed mutation effect."
            ),
        }, indent=2))

    elif args.command == "screen":
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
                "predicted_delta_vs_reference": score - ref_score,
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