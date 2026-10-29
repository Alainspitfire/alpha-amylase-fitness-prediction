# train_official.py

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_script(run_dir, filename, *arguments):
    command = [
        sys.executable,
        "-u",
        str(run_dir / filename),
        *map(str, arguments),
    ]

    log_path = run_dir / "logs" / f"{Path(filename).stem}.log"

    print(f"\n===== RUNNING {filename} =====", flush=True)

    with log_path.open("w") as log:
        with subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            status = process.wait()

    if status:
        raise subprocess.CalledProcessError(status, command)


def main():
    parser = argparse.ArgumentParser(
        description="Train the hybrid using the official FLIP2 split."
    )
    parser.add_argument("--out-dir", default="official_run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--esm-batch-size", type=int, default=2)

    args = parser.parse_args()

    if min(args.epochs, args.patience, args.esm_batch_size) < 1:
        parser.error("Epochs, patience, and batch size must be positive.")

    required = [
        "data.py",
        "model_mutation_effect.py",
        "train_rank.py",
        "train_esm_ridge.py",
        "blend_predictions.py",
        "predict_hybrid.py",
        "use_model.py",
    ]

    missing = [
        name for name in required
        if not (ROOT / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing source files:\n" + "\n".join(missing)
        )

    run_dir = Path(args.out_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} is not empty. "
            "Choose a new folder with --out-dir."
        )

    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is unavailable. Training will use CPU and may be slow.",
            flush=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir()

    copied = list(required)

    for optional in [
        "evaluate_official_split.py",
        "bootstrap_model_comparison.py",
    ]:
        if (ROOT / optional).is_file():
            copied.append(optional)

    for name in copied:
        shutil.copy2(ROOT / name, run_dir / name)

    shutil.copy2(
        Path(__file__).resolve(),
        run_dir / Path(__file__).name,
    )

    os.chdir(run_dir)

    from data import (
        download_dataset,
        split_dataset,
        build_reference,
    )

    # Download and verify the original dataset.
    raw = download_dataset()
    train, val, test = split_dataset(raw)

    if (len(train), len(val), len(test)) != (2403, 601, 702):
        raise ValueError("Unexpected official split sizes.")

    parts = {
        "train": train,
        "val": val,
        "test": test,
    }

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        if not set(parts[a].sequence).isdisjoint(
            parts[b].sequence
        ):
            raise ValueError(f"Sequence overlap: {a} and {b}")

    # Construct the reference using training sequences only.
    reference = build_reference(train)

    if len(reference) != 425:
        raise ValueError("Expected a 425-residue reference.")

    split_dir = run_dir / "official_split"
    split_dir.mkdir()

    # Legacy filenames are required by the existing trainers.
    # Their contents here are the official partitions.
    for name, frame in parts.items():
        frame.to_csv(
            split_dir / f"{name}_unseen.csv",
            index=False,
        )

    manifest = {
        "protocol": "Official FLIP2 alpha-amylase by-mutation",
        "reference": reference,
        "reference_definition": (
            "Consensus from official training sequences only"
        ),
        "train_size": len(train),
        "validation_size": len(val),
        "test_size": len(test),
        "filename_note": (
            "*_unseen.csv filenames are retained for "
            "compatibility with existing trainers."
        ),
    }

    (split_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(
        "\nOfficial split: "
        "2403 training / 601 validation / 702 test",
        flush=True,
    )

    # Train all three Transformer seeds.
    # Checkpoint selection uses validation MSE.
    run_script(
        run_dir,
        "train_rank.py",
        "--split-dir", "official_split",
        "--out-dir", "outputs_rank",
        "--epochs", args.epochs,
        "--patience", args.patience,
    )

    # Fit feature scaling and Ridge on training examples.
    # Ridge alpha is selected using validation Spearman.
    run_script(
        run_dir,
        "train_esm_ridge.py",
        "--split-dir", "official_split",
        "--out-dir", "outputs_esm_ridge",
        "--batch-size", args.esm_batch_size,
    )

    # Select the hybrid weight using validation Spearman.
    run_script(
        run_dir,
        "blend_predictions.py",
        "--rank-dir", "outputs_rank",
        "--esm-dir", "outputs_esm_ridge",
        "--out-dir", "outputs_blend",
    )

    import importlib.metadata

    versions = {
        "python": sys.version,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    }

    for name in [
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    ]:
        versions[name] = importlib.metadata.version(name)

    (run_dir / "runtime_versions.json").write_text(
        json.dumps(versions, indent=2)
    )

    checksums = {
        name: hashlib.sha256(
            (run_dir / name).read_bytes()
        ).hexdigest()
        for name in copied
    }

    (run_dir / "source_checksums.json").write_text(
        json.dumps(checksums, indent=2)
    )

    (run_dir / "training_arguments.json").write_text(
        json.dumps(vars(args), indent=2)
    )

    print("\n===== TRAINING COMPLETE =====")
    print("Saved run:", run_dir)
    print("The blend was selected using validation data.")
    print("Run evaluate_official_split.py for detailed evaluation.")


if __name__ == "__main__":
    main()