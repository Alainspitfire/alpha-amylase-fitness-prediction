# Alpha-Amylase Fitness Prediction

A proof-of-concept project for predicting assay-specific starch-stain removal activity (“fitness”) and ranking alpha-amylase variants.

The main application is **prioritizing candidates for experimental testing**. The repository includes training code, trained parameters, prediction tools, official partitions, and evaluation results.

[Run fitness prediction and ranking in Colab](PREDICTION_GUIDE.md)

## Models and Selection

- **Transformer:** Three mutation-aware Transformers trained with regression and pairwise ranking losses. Best numerical accuracy and top-10 selection among the models evaluated here; runs without ESM-2.
- **ESM–Ridge:** Ridge regression on frozen ESM-2 mean embedding differences and mutation count. A hybrid component and internal baseline.
- **Hybrid:** A validation-selected blend of **30% Transformer + 70% ESM–Ridge**. Highest observed overall ranking scores, but its ranking advantage over the Transformer was inconclusive under bootstrap resampling.

| Goal | Suggested option | Reason |
|---|---|---|
| Estimate numerical fitness | **Transformer** | Lowest test MAE and RMSE |
| Screen with fewer computational resources | **Transformer** | Does not load the 650M-parameter ESM backbone |
| Prioritize a small candidate shortlist | **Transformer** | Strongest observed top-10 and top-50 selection |
| Explore overall candidate ordering | **Hybrid** | Highest observed Spearman and pairwise accuracy |

Both options predict continuous fitness values; ranking sorts those predictions. Recommendations reflect this experiment and may not hold for new candidate collections.

Predictions can support candidate selection, variant comparison, and substitution screening. They are not probabilities of success or experimentally confirmed mutation effects.

## Dataset and Training

The project uses the FLIP2 alpha-amylase **by-mutation** dataset:

| Partition | Sequences | Purpose |
|---|---:|---|
| Training | 2,403 | Fit Transformer models, feature scaling, and Ridge coefficients |
| Validation | 601 | Select checkpoints, regularization, and blend weight |
| Test | 702 | Evaluate selected models |

The reference is a **consensus computed from training sequences only**, not necessarily the experimental wild type. Legacy filenames `train_unseen.csv`, `val_unseen.csv`, and `test_unseen.csv` contain the official partitions. The downloader verifies the dataset using an expected MD5 checksum.

Model settings:

- Transformer seeds: **42, 123, 2026**; checkpoints selected by validation MSE.
- Frozen ESM backbone: **`facebook/esm2_t33_650M_UR50D`**.
- ESM–Ridge features: mean-pooled embedding difference from the reference plus mutation count.
- ESM–Ridge alpha: **100**, selected by validation Spearman.
- Hybrid weight: selected by validation Spearman on a grid spaced by 0.1.
- Selected hybrid validation Spearman: **0.76783**.

```text
Predicted hybrid fitness = 0.30 × Transformer + 0.70 × ESM–Ridge
```

Inference uses saved configuration weights; retraining may select different weights.

## Official Test Results

Results cover **702 sequences**.

| Model | MSE ↓ | MAE ↓ | RMSE ↓ | Pearson ↑ | Spearman ↑ | Pairwise accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | **0.001148** | **0.022796** | **0.033888** | **0.775053** | 0.661013 | 90.32% |
| ESM–Ridge | 0.001795 | 0.034504 | 0.042364 | 0.654055 | 0.643054 | 88.38% |
| Hybrid | 0.001382 | 0.029744 | 0.037169 | 0.725605 | **0.673649** | **90.88%** |

Pairwise accuracy evaluates **128,045 pairs** with experimental fitness differences **≥0.03**; prediction ties receive half credit. It is not accuracy across all predictions or all sequence pairs.

### Published and Additional Baselines

FLIP2 reports Spearman of **0.648** for one-hot Ridge and **0.681** for one-hot + likelihoods Ridge. Our Transformer (0.661) and hybrid (0.674) fall between those published point estimates.

Source: [FLIP2, Table A4](https://flip.protein.properties/assets/FLIP_manuscipt.pdf).

Two additional internal baselines were evaluated on the same test sequences:

| Model | MSE ↓ | MAE ↓ | Spearman ↑ | Pairwise@0.03 ↑ | Top-10 hits ↑ |
|---|---:|---:|---:|---:|---:|
| One-hot Ridge | 0.001653 | 0.033171 | 0.648344 | 87.93% | 7 |
| One-hot + ESM-2 likelihood Ridge | 0.001645 | 0.033077 | 0.649485 | 88.02% | 7 |
| Transformer | **0.001148** | **0.022796** | 0.661013 | 90.32% | **9** |
| Hybrid | 0.001382 | 0.029744 | **0.673649** | **90.88%** | 8 |

Both models improved all listed metrics over these internal baselines. These comparisons are descriptive; the bootstrap intervals below do not cover the additional one-hot baselines or the direct Transformer–ESM–Ridge comparison.

One-hot Ridge selected **alpha = 10** using validation Spearman. The ESM-likelihood baseline used fixed **alpha = 10**, adding the frozen ESM-2 model’s unmasked mean residue log-probability to one-hot features. Its target scaler used training and validation labels; Ridge coefficients were fitted on training rows only. Errors were calculated on the original fitness scale.

This adapted baseline excludes CARP and Dayhoff features and does not reproduce the full published augmented Ridge model. Superiority over that published model in numerical accuracy, top-k selection, or computational cost has not been established.

To reproduce the additional baselines:

```bash
python train_onehot_ridge.py --root .
python train_onehot_esm_likelihood.py --root .
```

These generate `outputs_onehot_comparison/` and
`outputs_onehot_esm_likelihood/`, respectively. Archived summaries and
matched test predictions are under `alpha_amylase_baseline_results/`.

The baseline scripts run separately from `train_official.py`.

### Bootstrap

Paired bootstrap intervals use **2,000 resamples** of test sequences.

| Comparison | Metric | Hybrid minus baseline | 95% interval |
|---|---|---:|---:|
| Hybrid − Transformer | Spearman | +0.012636 | [−0.012537, +0.038389] |
| Hybrid − Transformer | Pairwise accuracy | +0.005623 | [−0.006186, +0.016913] |
| Hybrid − ESM–Ridge | Spearman | +0.030595 | [+0.018814, +0.042369] |
| Hybrid − ESM–Ridge | Pairwise accuracy | +0.025015 | [+0.018265, +0.032297] |

The hybrid’s ranking advantage over the Transformer is **inconclusive** because both intervals include zero. Its improvement over ESM–Ridge is supported under this analysis. The Transformer has lower MAE and MSE than the hybrid, with difference intervals excluding zero.

Related variants may violate the assumption of independent sequence resampling.

### Top-K Discovery

High-fitness hits belong to the **71 highest-fitness test variants**: `ceil(0.10 × 702)`.

| Model | Hits in top 10 | Top 25 | Top 50 |
|---|---:|---:|---:|
| Transformer | 9 | 17 | 27 |
| ESM–Ridge | 6 | 14 | 23 |
| Hybrid | 8 | 17 | 26 |

Higher overall Spearman does not guarantee better top-candidate selection. These top-k differences are descriptive; their uncertainty was not evaluated.

Earlier official-test results were inspected during development. This is an **exploratory follow-up experiment**, even though model selection uses validation data.

## Installation

Run from the repository root:

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

The recorded training environment used Python **3.13.16** and PyTorch **2.11.0+cu130**; see `runtime_versions.json`. Install a PyTorch build appropriate for your platform.

A CUDA GPU is recommended for training and ESM-based inference; scripts fall back to CPU. Internet access is required for initial dataset and ESM downloads. The pretrained backbone is not bundled.

`requirements.txt` is unpinned; the command above pins several dependencies. Exact numerical reproduction across environments is not guaranteed. Restart the notebook session after changing already imported packages.

## Repository Layout

```text
train_official.py               Official-split training entry point
train_rank.py                   Transformer ensemble training
train_esm_ridge.py              ESM embeddings and Ridge fitting
train_onehot_esm_likelihood.py  Additional likelihood-based baseline
blend_predictions.py           Validation-based hybrid selection
evaluate_official_split.py     Saved-prediction evaluation
bootstrap_model_comparison.py  Additional bootstrap tool
data.py                        Dataset download and encoding
model_mutation_effect.py       Transformer architecture
predict_hybrid.py              Hybrid loading and inference
train_onehot_ridge.py          One-hot Ridge baseline and comparison
use_model.py                   Prediction and ranking commands
requirements.txt
runtime_versions.json
PREDICTION_GUIDE.md

official_split/                Official partitions and reference
outputs_rank/                  Transformer checkpoints and predictions
outputs_esm_ridge/             Ridge parameters and predictions
outputs_blend/                 Hybrid configuration and predictions
outputs_official_test/         Metrics, intervals, and plots
alpha_amylase_baseline_results/ Internal one-hot baseline comparisons inspired by FLIP2
```

## Predict Fitness and Rank Candidates

Inputs must contain **425 standard amino-acid residues**, aligned to the saved reference. Insertions, deletions, and ambiguous symbols are unsupported.

Check required files:

```bash
python use_model.py setup --model transformer
python use_model.py setup --model hybrid
```

`setup` checks file presence, not prediction accuracy. Both options require all three Transformer checkpoints. Hybrid inference additionally requires the Ridge parameters, both model configurations, and downloaded ESM backbone.

### Predict One Sequence

Use a single-record FASTA:

```bash
python use_model.py fitness \
    --model transformer \
    --fasta sequence.fasta
```

Replace `transformer` with `hybrid` to use the hybrid.

### Rank Candidates

For CSV input with a `sequence` column:

```bash
python use_model.py rank \
    --model hybrid \
    --csv candidates.csv \
    --output ranked_predictions.csv
```

For multi-record FASTA:

```bash
python use_model.py rank \
    --model hybrid \
    --fasta candidates.fasta \
    --output ranked_predictions.csv
```

Use `--model transformer` for Transformer-only predictions.

**Rank 1 is the highest predicted fitness among submitted candidates.** Ties share ranks, such as 1, 1, 3. The model loads once per command.

Optional labels named `experimental_fitness` or `target` are detected automatically and used only for comparison. Label columns must contain finite numeric values in every row; omit both columns for unlabeled candidates. Custom columns can be specified with `--sequence-column` and `--fitness-column`.

A training sequence may be included as a reference alongside new candidates. Keep its known fitness separately if other candidates are unlabeled. Including a reference does not recalibrate predictions or provide an independent generalization test.

### Try the Official Test Data

```bash
python use_model.py rank \
    --model hybrid \
    --sample 20 \
    --seed 42 \
    --output test_sample_rankings.csv
```

This samples 20 bundled test sequences. Remove `--sample 20` to predict all 702. This demonstrates inference; it is not a new independent evaluation.

### Compare Variants or Screen Substitutions

Replace placeholders with complete sequences:

```bash
python use_model.py compare \
    --model transformer \
    --sequence-a "SEQUENCE_A" \
    --sequence-b "SEQUENCE_B"
```

Rank the 19 substitutions at reference position 100:

```bash
python use_model.py screen \
    --model transformer \
    --position 100 \
    --output position_100_screen.csv
```

Screening uses the saved consensus as the sequence background. Predicted differences require experimental confirmation.

## Train and Evaluate

Train a new run:

```bash
python train_official.py
```

This prepares the official partitions, constructs the reference, trains both components, and selects the blend. Artifacts are saved to `official_run/`; nonempty output directories are rejected.

Optional settings:

```bash
python train_official.py \
    --out-dir official_run_2 \
    --epochs 50 \
    --patience 10 \
    --esm-batch-size 2
```

When running either component trainer directly, pass `--split-dir official_split`; their default points to the older `mutation_holdout` directory.

Detailed evaluation is separate. For the supplied predictions:

```bash
python evaluate_official_split.py \
    --root . \
    --split-dir official_split \
    --out-dir outputs_official_test \
    --bootstraps 2000 \
    --seed 42
```

For a newly trained default run:

```bash
python official_run/evaluate_official_split.py \
    --root official_run \
    --split-dir official_split \
    --out-dir outputs_official_test \
    --bootstraps 2000 \
    --seed 42
```

The evaluator checks partitions, labels, and blend consistency, then generates metrics, subgroup results, top-k results, intervals, and plots. It evaluates **saved CSV predictions**, not freshly loaded checkpoints. Matching evaluation files are overwritten.

Predict using a newly trained run:

```bash
python use_model.py --root official_run \
    rank --model hybrid --csv candidates.csv
```

Training scripts print test metrics after fitting. Do not use them to tune subsequent settings.

## Limitations

- Results cover one alpha-amylase assay and a fixed sequence alignment.
- Generalization to other proteins, assay conditions, or substantially different sequences has not been established.
- Newly designed variants have not been prospectively validated.
- Predictions are continuous assay estimates, not probabilities of success.
- The hybrid does not consistently outperform the Transformer across objectives.
- Related sequences and execution environments can affect uncertainty estimates and numerical results.
- Official-test results were inspected during earlier development.
- This project does not implement Zhou et al.’s FSFP method; it is cited as related work.
- No state-of-the-art claim is made.

## Dataset Attribution

The dataset is distributed through FLIP2. Experimental measurements are not original contributions of this repository.

- [FLIP2 paper](https://proceedings.mlr.press/v306/didi26a.html)
- [Dataset archive](https://doi.org/10.5281/zenodo.18433203)
- [Official benchmark](https://flip.protein.properties/)

The original alpha-amylase measurements are from Floris van der Flier
et al., [“Enzyme Structure Correlates with Variant Effect Predictability”](https://doi.org/10.1016/j.csbj.2024.09.007).
This repository redistributes the FLIP2 by-mutation partitions as separate
CSV files and includes experimental labels alongside model predictions.

The original repository lists its non-software materials under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [FLIP2 amylase dataset note](https://zenodo.org/records/18433203/files/amylase/README.md?download=1)
lists MIT. This repository’s MIT license applies to the project code.

## AI-Assisted Development

ChatGPT assisted with code development, debugging, evaluation workflows, and documentation. Reported performance values come from computational evaluations of the models.

## Reference

Dallago, Christian, et al. "FLIP: Benchmark tasks in fitness landscape inference for proteins." bioRxiv (2021): 2021-11.

Didi, Kieran, et al. "FLIP2: Expanding protein fitness landscape benchmarks for real-world machine learning applications." bioRxiv (2026): 2026-02.

Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.

Zhou, Ziyi, et al. "Enhancing efficiency of protein language models with minimal wet-lab data through few-shot learning." Nature Communications 15.1 (2024): 5566.
