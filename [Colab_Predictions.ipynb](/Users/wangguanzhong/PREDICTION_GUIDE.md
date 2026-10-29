## Run Fitness Prediction and Ranking in Google Colab

This guide downloads the code and trained models directly from the public GitHub repository. No retraining is required.

### Before starting

1. Open a new [Google Colab notebook](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → GPU**.
3. Run the two cells below in order.

Input sequences must contain **425 standard amino-acid residues**, aligned to the model reference. Insertions, deletions, gaps, and ambiguous amino-acid symbols are unsupported.

### Cell 1: Install dependencies and download the model

```python
%pip install -q --only-binary=:all: "transformers==4.57.6" "tokenizers==0.22.2" "huggingface-hub==0.36.2" "pandas==2.2.3" "numpy==2.2.6" requests safetensors

import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
import torch

assert torch.cuda.is_available(), (
    "Select Runtime → Change runtime type → GPU."
)

repo = "Alainspitfire/alpha-amylase-fitness-prediction"

# Resolve main to a specific commit for reproducibility.
response = requests.get(
    f"https://api.github.com/repos/{repo}/commits/main",
    timeout=60,
)

if response.status_code != 200:
    raise RuntimeError(
        f"GitHub returned HTTP {response.status_code}. "
        "This example requires the repository to be public."
    )

commit = response.json()["sha"]

response = requests.get(
    f"https://api.github.com/repos/{repo}/zipball/{commit}",
    timeout=180,
)

if response.status_code != 200:
    raise RuntimeError(
        f"Download failed: HTTP {response.status_code}"
    )

destination = Path(
    tempfile.mkdtemp(prefix="amylase_", dir="/content")
)

with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    for name in archive.namelist():
        target = (destination / name).resolve()
        assert target.is_relative_to(destination.resolve())

    archive.extractall(destination)

matches = [
    path.parent
    for path in destination.rglob("use_model.py")
    if (
        path.parent / "outputs_blend/ensemble_config.json"
    ).is_file()
]

assert len(matches) == 1, (
    f"Expected one model directory; found {len(matches)}."
)

project = matches[0]

assert "def rank_sequences(" in (
    project / "use_model.py"
).read_text(), (
    "The repository requires the updated use_model.py "
    "containing the rank command."
)

os.chdir(project)

print("GitHub commit:", commit)
print("GPU:", torch.cuda.get_device_name(0))
print("Ready for prediction.")
```

### Cell 2: Choose input, predict fitness, and rank candidates

The cell asks for an input method and model. It passes your answers to the command-line options defined with `argparse` in `use_model.py`.

```python
import csv
import pandas as pd
from IPython.display import display
from google.colab import files

print("Choose input:")
print("1. Paste sequences separated by commas")
print("2. Upload a CSV")
print("3. Upload a multi-record FASTA")
print("4. Try a random sample of the official test set")

choice = input("Input method [1]: ").strip() or "1"

model = (
    input("Model: hybrid or transformer [hybrid]: ")
    .strip()
    .lower()
    or "hybrid"
)

if model not in {"hybrid", "transformer"}:
    raise ValueError("Choose hybrid or transformer.")

output = Path("/content/ranked_predictions.csv")

command = [
    sys.executable,
    "-u",
    str(project / "use_model.py"),
    "--root", str(project),
    "rank",
    "--model", model,
    "--output", str(output),
]

if choice == "1":
    text = input(
        "Paste amino-acid sequences separated by commas: "
    )

    sequences = [
        "".join(sequence.split()).upper()
        for sequence in text.split(",")
        if sequence.strip()
    ]

    if not sequences:
        raise ValueError("Enter at least one sequence.")

    input_dir = Path(
        tempfile.mkdtemp(prefix="candidates_", dir="/content")
    )
    input_file = input_dir / "candidates.csv"

    with input_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence"])

        writer.writerows(
            (f"candidate_{number}", sequence)
            for number, sequence in enumerate(sequences, 1)
        )

    command += ["--csv", str(input_file)]

elif choice in {"2", "3"}:
    uploaded = files.upload()

    if len(uploaded) != 1:
        raise ValueError("Upload exactly one input file.")

    filename = next(iter(uploaded))

    input_dir = Path(
        tempfile.mkdtemp(prefix="candidates_", dir="/content")
    )
    input_file = input_dir / Path(filename).name
    input_file.write_bytes(uploaded[filename])

    if choice == "2":
        sequence_column = (
            input("Sequence column name [sequence]: ").strip()
            or "sequence"
        )

        command += [
            "--csv", str(input_file),
            "--sequence-column", sequence_column,
        ]

        fitness_column = input(
            "Measured fitness column "
            "(blank = auto-detect or none): "
        ).strip()

        if fitness_column:
            command += ["--fitness-column", fitness_column]

    else:
        command += ["--fasta", str(input_file)]

elif choice == "4":
    count = int(
        input("Number of test sequences [20]: ").strip()
        or "20"
    )

    command += [
        "--sample", str(count),
        "--seed", "42",
    ]

else:
    raise ValueError("Choose 1, 2, 3, or 4.")

# Loads the selected model once for all candidates.
subprocess.run(
    command,
    cwd=project,
    check=True,
)

results = pd.read_csv(output)

display(results.drop(columns="sequence"))

print("Predictions generated from GitHub commit:", commit)

files.download(str(output))
```

### Input formats

**CSV**

The default sequence column is `sequence`. An optional `id` column identifies candidates.

Experimental fitness is optional. The script automatically detects `experimental_fitness` or `target`, or you can specify another column at the prompt. If included, measurements must be finite numbers for every row.

**FASTA**

Provide one or more sequences, each preceded by a header beginning with `>`.

### Understanding the output

The downloaded `ranked_predictions.csv` contains:

| Column | Meaning |
|---|---|
| `predicted_rank` | Rank among the supplied candidates; 1 means highest predicted fitness |
| `predicted_fitness` | Predicted fitness on the dataset's experimental target scale |
| `transformer_fitness` | Transformer component prediction when using the hybrid |
| `esm_ridge_fitness` | ESM–Ridge component prediction when using the hybrid |
| `mutation_count` | Differences from the saved consensus reference |
| `experimental_fitness` | Supplied measurement, when available |
| `experimental_rank` | Rank based on supplied measurements, when available |
| `absolute_error` | Absolute prediction error, when measurements are available |

The CSV also includes candidate identifiers, sequences, and source row numbers.

Ties share ranks—for example, **1, 1, 3**. Experimental measurements are used only for comparison; they do not determine predicted rankings.

### Command-line usage

The same operations can be run locally after installing dependencies:

```bash
# Predict fitness and rank CSV candidates.
python use_model.py rank --csv candidates.csv

# Predict fitness and rank FASTA candidates.
python use_model.py rank --fasta candidates.fasta

# Try 20 randomly selected test sequences.
python use_model.py rank --sample 20 --seed 42

# Use the Transformer ensemble.
python use_model.py rank --model transformer --csv candidates.csv

# Predict fitness for one FASTA sequence.
python use_model.py fitness --model hybrid --fasta sequence.fasta

# Show available ranking options.
python use_model.py rank --help
```

### Interpretation

The supplied hybrid combines **30% Transformer ensemble** and **70% ESM–Ridge**, selected using validation Spearman correlation.

Its predictions support a proof of concept for prioritizing alpha-amylase variants. They are not experimentally confirmed outcomes or validated predictions for unrelated proteins.

Ranking a test-set sample demonstrates inference; it is not new independent validation. New candidate predictions require experimental testing.
