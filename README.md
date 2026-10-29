# Alpha-Amylase Fitness Prediction

A proof-of-concept project for ranking alpha-amylase variants by experimental fitness using a mutation-aware Transformer ensemble, ESM-2 embeddings with Ridge regression, and a validation-selected hybrid.

The repository includes training code, trained model parameters, inference tools, official split records, predictions, and evaluation results.

## Models

### Mutation-Aware Transformer

An ensemble of three Transformers trained with seeds **42, 123, and 2026**.

Each model uses amino-acid, positional, and mutation representations, followed by global and mutation-aware pooling. Training combines mean squared error with a pairwise logistic ranking loss.

Checkpoints are selected using validation MSE. The three models' fitness predictions are averaged.

### ESM–Ridge

This model uses the frozen pretrained backbone:

```text
facebook/esm2_t33_650M_UR50D
```

Residue embeddings are mean-pooled, excluding special tokens. Features consist of the difference between the variant and reference embeddings, plus mutation count.

Feature scaling and Ridge coefficients are fitted on the training partition only. Ridge regularization is selected using validation Spearman correlation.

The supplied Ridge model uses **alpha = 100**.

### Hybrid

The final supplied model uses:

```text
Predicted fitness = 0.30 × Transformer + 0.70 × ESM–Ridge
```

The Transformer weight was selected from 0.0 to 1.0 in increments of 0.1 using validation Spearman correlation.

The selected hybrid achieved validation Spearman **0.76783**.

Retraining may select a different blend weight. Inference reads the saved configuration rather than assuming fixed weights.

## Dataset and Training Protocol

This project uses the FLIP2 alpha-amylase **by-mutation** dataset and its official partitions:

| Partition | Sequences | Purpose |
|---|---:|---|
| Training | 2,403 | Fit Transformer models, feature scaling, and Ridge coefficients |
| Validation | 601 | Select checkpoints, Ridge regularization, and blend weight |
| Test | 702 | Evaluate the selected models |

The reference sequence is a consensus computed from official training sequences only.

The split files retain the names `train_unseen.csv`, `val_unseen.csv`, and `test_unseen.csv` for compatibility with the training scripts. In this version, they contain the **official partitions**, not the earlier custom holdout.

The dataset downloader verifies the downloaded archive using an expected MD5 checksum.

## Official Test Results

| Model | MSE ↓ | MAE ↓ | RMSE ↓ | Pearson ↑ | Spearman ↑ | Pairwise accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | **0.001148** | **0.022796** | **0.033888** | **0.775053** | 0.661013 | 90.32% |
| ESM–Ridge | 0.001795 | 0.034504 | 0.042364 | 0.654055 | 0.643054 | 88.38% |
| Hybrid | 0.001382 | 0.029744 | 0.037169 | 0.725605 | **0.673649** | **90.88%** |

Pairwise accuracy is calculated over the **128,045 test pairs** whose experimental fitness differs by at least **0.03**. Prediction ties receive half credit.

The hybrid has the highest observed Spearman correlation and pairwise accuracy. The Transformer has lower prediction error and higher Pearson correlation.

### Bootstrap Comparisons

Paired bootstrap intervals use 2,000 resamples of test sequences.

| Comparison | Metric | Hybrid minus baseline | 95% interval |
|---|---|---:|---:|
| Hybrid − Transformer | Spearman | +0.012636 | [−0.012537, +0.038389] |
| Hybrid − Transformer | Pairwise accuracy | +0.005623 | [−0.006186, +0.016913] |
| Hybrid − ESM–Ridge | Spearman | +0.030595 | [+0.018814, +0.042369] |
| Hybrid − ESM–Ridge | Pairwise accuracy | +0.025015 | [+0.018265, +0.032297] |

The hybrid's ranking advantage over the Transformer is inconclusive under this bootstrap procedure. Its ranking improvement over ESM–Ridge is supported by intervals excluding zero.

The Transformer also has lower MAE and MSE than the hybrid, with bootstrap difference intervals excluding zero in the Transformer's favor.

Related variants may violate the assumption that sequences can be resampled independently.

### Top-K Discovery

High-fitness variants are defined as the top `ceil(0.10 × 702) = 71` test variants by experimental fitness.

| Model | High-fitness hits in top 10 | In top 25 | In top 50 |
|---|---:|---:|---:|
| Transformer | 9 | 17 | 27 |
| ESM–Ridge | 6 | 14 | 23 |
| Hybrid | 8 | 17 | 26 |

These are descriptive results; uncertainty for top-k differences was not evaluated.

### Interpretation

The hybrid's strongest demonstrated use is prioritizing alpha-amylase variants by predicted fitness. The Transformer is stronger at predicting numerical fitness values in this experiment.

These results support a proof of concept, not a claim of state-of-the-art performance or superiority across all metrics.

Earlier official-test results were inspected during project development. This run should therefore be reported as a **follow-up experiment**, even though the training code selects parameters using validation data.

## Installation

Run commands from the repository root.

```bash
python -m pip install -r requirements.txt \
    "transformers==4.57.6" \
    "tokenizers==0.22.2" \
    "huggingface-hub==0.36.2" \
    "pandas==2.2.3" \
    "numpy==2.2.6" \
    "scipy==1.16.3" \
    "scikit-learn==1.6.1" \
    "matplotlib==3.10.0"
```

The recorded training environment used Python **3.13.16** and PyTorch **2.11.0+cu130**. See `runtime_versions.json` for the complete recorded environment.

Install a PyTorch build appropriate for your platform and CUDA environment. A CUDA GPU is recommended for training and ESM inference. The scripts use CPU when CUDA is unavailable.

Internet access is required to download the dataset and pretrained ESM backbone on first use. The pretrained backbone itself is not bundled with the repository.

## Repository Layout

```text
train_official.py
train_rank.py
train_esm_ridge.py
blend_predictions.py
evaluate_official_split.py
bootstrap_model_comparison.py
data.py
model_mutation_effect.py
predict_hybrid.py
use_model.py
requirements.txt
runtime_versions.json

official_split/
    split_manifest.json
    train_unseen.csv
    val_unseen.csv
    test_unseen.csv

outputs_rank/
    rank_seed_42.pt
    rank_seed_123.pt
    rank_seed_2026.pt
    train_predictions.csv
    val_predictions.csv
    test_predictions.csv

outputs_esm_ridge/
    configuration.json
    ridge_head.npz
    train_predictions.csv
    val_predictions.csv
    test_predictions.csv

outputs_blend/
    ensemble_config.json
    val_predictions.csv
    test_predictions.csv

outputs_official_test/
    official_test_metrics.csv
    official_test_predictions.csv
    official_test_topk.csv
    all_split_metrics.csv
    subgroup_metrics.csv
    paired_bootstrap_intervals.csv
    train_matched_predictions.csv
    val_matched_predictions.csv
    test_matched_predictions.csv
    model_comparison.png
    prediction_scatter.png
    evaluation_protocol.json
```

## Use the Supplied Models

All three Transformer checkpoints are required for the final ensemble.

Hybrid inference additionally requires `ridge_head.npz`, both model configuration files, and access to the pretrained ESM backbone.

Check that the required files are present:

```bash
python use_model.py setup --model hybrid
```

Predict fitness from a FASTA file containing one sequence:

```bash
python use_model.py fitness --model hybrid --fasta sequence.fasta
```

Use the Transformer ensemble alone:

```bash
python use_model.py fitness --model transformer --fasta sequence.fasta
```

Input sequences must contain **425 standard amino-acid residues** and use the same residue alignment as the reference. Insertions and deletions are unsupported.

Predictions are on the dataset's experimental target scale.

The `setup` command checks file presence; it does not verify prediction accuracy.

## Train a New Model

```bash
python train_official.py
```

This command:

1. Downloads and verifies the dataset.
2. Saves the official training, validation, and test partitions.
3. Constructs a reference from training sequences.
4. Trains the three Transformer seeds.
5. Extracts frozen ESM-2 embeddings and fits Ridge regression.
6. Selects the hybrid blend using validation Spearman.
7. Saves model artifacts, predictions, logs, and environment metadata.

The new run is written to `official_run/`, preserving the supplied results. The script refuses to use a nonempty output directory.

To customize the run:

```bash
python train_official.py \
    --out-dir official_run_2 \
    --epochs 50 \
    --patience 10 \
    --esm-batch-size 2
```

A smaller ESM batch size may help if GPU memory is insufficient.

The training scripts print test metrics after fitting. Do not use these printed values to tune subsequent settings.

## Evaluate the Supplied Predictions

```bash
python evaluate_official_split.py \
    --root . \
    --split-dir official_split \
    --out-dir outputs_official_test \
    --bootstraps 2000 \
    --seed 42
```

The evaluator:

- Checks partition sizes and sequence overlap.
- Aligns predictions by sequence and checks experimental labels.
- Checks that hybrid predictions match the saved blend weights.
- Calculates training, validation, and test metrics.
- Calculates test mutation-count subgroup and top-k results.
- Computes paired bootstrap intervals.
- Saves plots and evaluation metadata.

It evaluates saved prediction CSVs. It does not reload checkpoints or rerun ESM inference.

Running this command overwrites evaluation files with matching names.

## Evaluate a Newly Trained Run

After `python train_official.py` completes:

```bash
python official_run/evaluate_official_split.py \
    --root official_run \
    --split-dir official_split \
    --out-dir outputs_official_test \
    --bootstraps 2000 \
    --seed 42
```

Results are saved inside:

```text
official_run/outputs_official_test/
```

To use the newly trained hybrid:

```bash
python use_model.py --root official_run \
    fitness --model hybrid --fasta sequence.fasta
```

## Limitations

- Evaluation covers one alpha-amylase dataset and a fixed sequence alignment.
- Generalization to other proteins or substantially different sequences has not been established.
- Predicted improvements require experimental validation.
- The hybrid's observed ranking advantage over the Transformer is not conclusive under the reported bootstrap.
- The Transformer performs better on numerical prediction error and some top-k measures.
- Hardware and dependency versions can produce numerical differences.
- Official-test results were inspected during earlier development.
- This project does not implement the FSFP method described by Zhou et al.; that paper is cited as related work.

## Dataset Attribution

The dataset is distributed through FLIP2. The experimental measurements are not original contributions of this repository.

- [FLIP2 paper](https://proceedings.mlr.press/v306/didi26a.html)
- [Dataset archive](https://doi.org/10.5281/zenodo.18433203)
- [Official benchmark](https://flip.protein.properties/)

## AI-Assisted Development

ChatGPT assisted with code development, debugging, evaluation workflows, and documentation. Reported performance values come from computational evaluations of the models.

## Reference

Dallago, Christian, et al. "FLIP: Benchmark tasks in fitness landscape inference for proteins." bioRxiv (2021): 2021-11.

Didi, Kieran, et al. "FLIP2: Expanding protein fitness landscape benchmarks for real-world machine learning applications." bioRxiv (2026): 2026-02.

Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.

Zhou, Ziyi, et al. "Enhancing efficiency of protein language models with minimal wet-lab data through few-shot learning." Nature Communications 15.1 (2024): 5566.
