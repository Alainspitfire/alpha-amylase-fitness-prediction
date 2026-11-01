# Alpha-Amylase Fitness Prediction

A proof-of-concept project for predicting assay-specific starch-stain removal activity (“fitness”) and ranking alpha-amylase variants.

The main application is **prioritizing candidate variants for experimental testing**. The project provides two prediction options: a lightweight Transformer ensemble and a hybrid combining that ensemble with ESM–Ridge.

The repository includes training code, trained parameters, inference tools, official split records, and evaluation results.

[Run fitness prediction and ranking in Colab](PREDICTION_GUIDE.md)

## Models

- **Transformer:** An ensemble of three mutation-aware Transformers trained with regression and pairwise ranking losses. It provides the strongest numerical fitness predictions in this experiment and runs without ESM-2.
- **ESM–Ridge:** Ridge regression on frozen ESM-2 embedding differences and mutation count. It is a component of the hybrid and an evaluation baseline.
- **Hybrid:** A validation-selected combination of **30% Transformer + 70% ESM–Ridge**. It achieved the highest observed overall ranking scores, although its advantage over the Transformer was inconclusive under bootstrap resampling.

Both user-facing options, `transformer` and `hybrid`, predict numerical fitness values. The `rank` command sorts those values to rank candidates.

## Which Model Should I Select?

| Your goal | Suggested option | Reason |
|---|---|---|
| Predict numerical fitness values | **Transformer** | Lower MAE and RMSE on the official test |
| Run predictions with fewer computational resources | **Transformer** | Does not load the 650M-parameter ESM backbone |
| Obtain an initial candidate shortlist | **Transformer** | Strong top-k results and lower computational cost |
| Explore overall ranking across a candidate collection | **Hybrid** | Highest observed Spearman and pairwise accuracy |
| Compare prediction strategies before experiments | **Both** | Examine differences in predicted fitness and candidate ordering |

**Practical starting point:** use the Transformer for numerical prediction and an efficient first screening pass. Use the hybrid when overall ordering is your primary objective and the additional ESM computation is acceptable.

The hybrid is not consistently better at selecting the very best few candidates: the Transformer found more high-fitness variants in the top 10 and top 50 in this test.

These recommendations describe this experiment's results. They do not guarantee the same ordering of model performance on new candidate collections.

## What Are the Predictions Useful For?

### Candidate prioritization

Rank a collection of compatible alpha-amylase variants and shortlist candidates for activity assays. This can help allocate experimental effort when testing every candidate is impractical.

### Approximate fitness estimation

Estimate fitness on the dataset's experimental target scale. The Transformer is the stronger option for this task in the reported evaluation.

These outputs are continuous estimates, not probabilities of success.

### Comparing variants

Compare predicted fitness between two sequences to identify which the model expects to perform better.

A predicted difference is a screening signal. It does not establish a causal mutation effect or confirm that an improvement will occur experimentally.

### Exploring substitutions

The `screen` command ranks the 19 possible substitutions at one position in the saved reference sequence. This can generate hypotheses for subsequent experiments.

Screening an arbitrary substitution is not equivalent to experimentally validating it.

### Scope

The supported application is prediction within the alpha-amylase sequence and assay setting represented by this dataset. Performance on unrelated proteins, different assay conditions, or substantially different sequences has not been established.

## Dataset and Training Protocol

This project uses the FLIP2 alpha-amylase **by-mutation** dataset and its official partitions:

| Partition | Sequences | Purpose |
|---|---:|---|
| Training | 2,403 | Fit Transformer models, feature scaling, and Ridge coefficients |
| Validation | 601 | Select checkpoints, Ridge regularization, and blend weight |
| Test | 702 | Evaluate the selected models |

The reference is a consensus computed from official training sequences only.

For compatibility with the trainers, the split filenames remain `train_unseen.csv`, `val_unseen.csv`, and `test_unseen.csv`. Their contents are the **official partitions**.

The dataset downloader verifies the downloaded archive using an expected MD5 checksum.

### Model Selection

- Transformer seeds: **42, 123, and 2026**
- Transformer checkpoint selection: **validation MSE**
- ESM backbone: **`facebook/esm2_t33_650M_UR50D`**, kept frozen
- Ridge features: mean-pooled embedding difference from the reference, plus mutation count
- Ridge regularization: **alpha = 100**, selected using validation Spearman
- Hybrid weight: selected using validation Spearman from weights spaced by 0.1
- Selected hybrid validation Spearman: **0.76783**

The supplied hybrid uses:

```text
Predicted fitness = 0.30 × Transformer + 0.70 × ESM–Ridge
```

Inference reads the saved weights from the configuration. Retraining may select different weights.

## Official Test Results

| Model | MSE ↓ | MAE ↓ | RMSE ↓ | Pearson ↑ | Spearman ↑ | Pairwise accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | **0.001148** | **0.022796** | **0.033888** | **0.775053** | 0.661013 | 90.32% |
| ESM–Ridge | 0.001795 | 0.034504 | 0.042364 | 0.654055 | 0.643054 | 88.38% |
| Hybrid | 0.001382 | 0.029744 | 0.037169 | 0.725605 | **0.673649** | **90.88%** |

Pairwise accuracy considers the **128,045 test pairs** whose experimental fitness differs by at least **0.03**. Prediction ties receive half credit.

The hybrid's 90.88% pairwise accuracy therefore describes pairs with sufficiently separated experimental fitness. It is not an accuracy percentage for all predictions or all sequence pairs.

### Comparison with Published and Internal Baselines

FLIP2 reports Spearman correlations of 0.648 for Ridge using one-hot
features and 0.681 for Ridge using one-hot features plus likelihoods
on the alpha-amylase by-mutation benchmark. This project's Transformer
achieved 0.661 and its hybrid achieved 0.674, placing both point estimates
between those published baselines.

Within this project, the Transformer also outperformed an internally
trained ESM–Ridge baseline on observed Spearman correlation
(0.661 versus 0.643), pairwise accuracy at a minimum experimental fitness
gap of 0.03 (90.32% versus 88.38%), and high-fitness hits among the top
10 predictions (9 versus 6). This ESM–Ridge baseline is not a reproduction
of the published FLIP2 Ridge models. Statistical uncertainty for the
direct Transformer–ESM–Ridge differences was not evaluated.

The top-10 results support the Transformer's potential usefulness for
candidate prioritization despite its slightly lower Spearman than the strongest
listed published baseline. However, superiority over published models
in top-k selection, numerical accuracy, or computational cost has not
been established. No state-of-the-art performance claim is made.

Source: [FLIP2, Table A4](https://flip.protein.properties/assets/FLIP_manuscipt.pdf).

### Additional One-Hot Baseline Comparisons

Two additional baselines were evaluated on the same 702 test sequences:

| Model | MSE ↓ | MAE ↓ | Spearman ↑ | Pairwise@0.03 ↑ | Top-10 hits ↑ |
|---|---:|---:|---:|---:|---:|
| One-hot Ridge | 0.001653 | 0.033171 | 0.648344 | 87.93% | 7 |
| One-hot + ESM-2 likelihood Ridge | 0.001645 | 0.033077 | 0.649485 | 88.02% | 7 |
| Transformer | **0.001148** | **0.022796** | 0.661013 | 90.32% | **9** |
| Hybrid | 0.001382 | 0.029744 | **0.673649** | **90.88%** | 8 |

Both the Transformer and hybrid improved all listed metrics over these
baselines. These are descriptive comparisons; the bootstrap intervals
below do not cover the additional one-hot baselines.

One-hot Ridge selected alpha = 10 using validation Spearman. The
one-hot + ESM-2 likelihood baseline used fixed alpha = 10 and appended
the frozen ESM-2 model's unmasked mean residue log-probability to the
one-hot features. Its target scaler used training and validation labels,
while Ridge coefficients were fitted on training rows only. Errors
were evaluated on the original fitness scale.

This adapted baseline excludes CARP and Dayhoff scores and does not
reproduce the full published FLIP2 augmented Ridge model.

To reproduce the ESM-2 likelihood comparison:

```bash
python train_onehot_esm_likelihood.py --root .
```

Results are saved in `outputs_onehot_esm_likelihood/`.

### Bootstrap Comparisons

Paired bootstrap intervals use 2,000 resamples of test sequences.

| Comparison | Metric | Hybrid minus baseline | 95% interval |
|---|---|---:|---:|
| Hybrid − Transformer | Spearman | +0.012636 | [−0.012537, +0.038389] |
| Hybrid − Transformer | Pairwise accuracy | +0.005623 | [−0.006186, +0.016913] |
| Hybrid − ESM–Ridge | Spearman | +0.030595 | [+0.018814, +0.042369] |
| Hybrid − ESM–Ridge | Pairwise accuracy | +0.025015 | [+0.018265, +0.032297] |

The hybrid's ranking advantage over the Transformer is **inconclusive** because both difference intervals include zero. Its ranking improvement over ESM–Ridge is supported by intervals excluding zero.

The Transformer has lower MAE and MSE than the hybrid, with bootstrap difference intervals excluding zero in the Transformer's favor.

Related variants may violate the assumption that sequences can be resampled independently.

### Top-K Discovery

High-fitness variants are defined as the top `ceil(0.10 × 702) = 71` test variants by experimental fitness.

| Model | High-fitness hits in top 10 | In top 25 | In top 50 |
|---|---:|---:|---:|
| Transformer | 9 | 17 | 27 |
| ESM–Ridge | 6 | 14 | 23 |
| Hybrid | 8 | 17 | 26 |

These results show that overall ranking performance and top-candidate discovery are different objectives. A higher Spearman correlation does not guarantee a better top-10 shortlist.

Top-k comparisons are descriptive; uncertainty for these differences was not evaluated.

### Interpretation

The results support a proof of concept for alpha-amylase candidate prioritization.

- The Transformer is stronger at predicting numerical fitness in this experiment.
- The hybrid has the highest observed overall ranking scores.
- The hybrid's ranking advantage over the Transformer is not statistically conclusive under the reported bootstrap.
- Neither model has been prospectively validated on newly designed candidates.

Earlier official-test results were inspected during development. This run is therefore reported as a **follow-up experiment**, even though the selection code uses validation data.

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

The recorded training environment used Python **3.13.16** and PyTorch **2.11.0+cu130**. See `runtime_versions.json` for the recorded environment.

Install a PyTorch build appropriate for your platform and CUDA environment. A CUDA GPU is recommended for training and hybrid inference. The scripts use CPU when CUDA is unavailable.

Internet access is required for the initial dataset and pretrained ESM backbone downloads. The pretrained backbone is not bundled with the repository.

## Repository Layout

```text
train_official.py              Official-split training entry point
train_rank.py                  Transformer ensemble training
train_esm_ridge.py             ESM feature extraction and Ridge fitting
blend_predictions.py          Validation-based hybrid selection
evaluate_official_split.py    Saved-prediction evaluation
bootstrap_model_comparison.py Additional bootstrap comparison tool
data.py                       Dataset download and sequence encoding
model_mutation_effect.py      Transformer architecture
predict_hybrid.py             Hybrid model loading and inference
use_model.py                  User-facing prediction commands
train_onehot_esm_likelihood.py  One-hot + ESM-2 likelihood Ridge comparison
requirements.txt
runtime_versions.json
PREDICTION_GUIDE.md

official_split/               Official partitions and reference manifest
outputs_rank/                 Transformer checkpoints and predictions
outputs_esm_ridge/            Ridge parameters, configuration, predictions
outputs_blend/                Hybrid configuration and predictions
outputs_official_test/        Metrics, predictions, intervals, and plots
outputs_onehot_comparison/     One-hot Ridge comparison results
outputs_onehot_esm_likelihood/ One-hot + ESM-2 likelihood results
```

## Predict Fitness and Rank Candidates

Input sequences must contain **425 standard amino-acid residues**, aligned to the saved reference. Insertions, deletions, and ambiguous symbols are unsupported.

### Check the Model Files

```bash
python use_model.py setup --model transformer
python use_model.py setup --model hybrid
```

The setup command checks file presence; it does not verify prediction accuracy.

All three Transformer checkpoints are required. Hybrid inference additionally requires `ridge_head.npz`, both model configurations, and the pretrained ESM backbone.

### Predict One Sequence

For numerical fitness prediction with the Transformer:

```bash
python use_model.py fitness \
    --model transformer \
    --fasta sequence.fasta
```

For hybrid prediction:

```bash
python use_model.py fitness \
    --model hybrid \
    --fasta sequence.fasta
```

Each FASTA file above must contain exactly one sequence.

### Rank Multiple Candidates

Use a CSV containing a `sequence` column:

```bash
python use_model.py rank \
    --model transformer \
    --csv candidates.csv \
    --output ranked_predictions.csv
```

Use the hybrid:

```bash
python use_model.py rank \
    --model hybrid \
    --csv candidates.csv \
    --output ranked_predictions.csv
```

Multi-record FASTA input is also supported:

```bash
python use_model.py rank \
    --model hybrid \
    --fasta candidates.fasta \
    --output ranked_predictions.csv
```

The model loads once for the complete command. Output includes numerical fitness predictions and ranks.

**Rank 1 means highest predicted fitness within the submitted candidates.** Ties share ranks, such as 1, 1, 3.

Optional CSV labels named `experimental_fitness` or `target` are detected
automatically and used only for comparison. If either column is present,
every row must contain a finite numeric value. For unlabeled candidates,
omit both columns entirely.

A sequence from the training data may be included as a reference alongside
new candidates. Keep its known experimental fitness separately when the
other candidates are unlabeled. Including a reference does not recalibrate
the model, and its prediction is not an independent test of generalization.

Custom columns can be specified using `--sequence-column` and
`--fitness-column`.

### Try the Official Test Data

```bash
python use_model.py rank \
    --model hybrid \
    --sample 20 \
    --seed 42 \
    --output test_sample_rankings.csv
```

This randomly selects 20 sequences from the bundled official test partition. Remove `--sample 20` to predict and rank all 702 sequences.

This is an inference demonstration, not a new independent evaluation.

### Compare Two Variants

Replace the placeholders with complete amino-acid sequences:

```bash
python use_model.py compare \
    --model transformer \
    --sequence-a "SEQUENCE_A" \
    --sequence-b "SEQUENCE_B"
```

### Screen One Position

For example, rank the 19 substitutions at reference position 100:

```bash
python use_model.py screen \
    --model transformer \
    --position 100 \
    --output position_100_screen.csv
```

This screen uses the saved reference sequence as the background.

## Train a New Model

```bash
python train_official.py
```

This prepares the official partitions, constructs the reference, trains both components, selects the blend, and saves artifacts and metadata.

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

The training scripts print test metrics after fitting. Do not use those values to tune subsequent settings.

### Reproduction Notes

Use `train_official.py` as the training entry point. When running
`train_rank.py` or `train_esm_ridge.py` directly, explicitly pass
`--split-dir official_split`; their default points to the older
`mutation_holdout` directory.

Detailed evaluation is a separate step after training, as shown below.

`requirements.txt` contains unpinned dependencies. The installation command
above pins several packages, while `runtime_versions.json` records the
original environment. Exact numerical reproduction across environments
is not guaranteed.

## Evaluate the Supplied Predictions

```bash
python evaluate_official_split.py \
    --root . \
    --split-dir official_split \
    --out-dir outputs_official_test \
    --bootstraps 2000 \
    --seed 42
```

The evaluator checks split sizes, sequence overlap, labels, and blend consistency. It saves regression and ranking metrics, subgroup results, top-k results, paired bootstrap intervals, and plots.

It evaluates **saved prediction CSVs**. It does not reload checkpoints or rerun inference.

Evaluation files with matching names are overwritten.

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

To use the newly trained models:

```bash
python use_model.py --root official_run \
    rank --model hybrid --csv candidates.csv
```

## Limitations

- Evaluation covers one alpha-amylase dataset and a fixed sequence alignment.
- Generalization to other proteins, assay conditions, or substantially different sequences has not been established.
- Numerical predictions and mutation comparisons require experimental confirmation.
- Predicted fitness is not a probability of experimental success.
- The hybrid does not consistently outperform the Transformer across metrics or screening budgets.
- Related sequences can affect bootstrap uncertainty estimates.
- Hardware and dependency versions can produce numerical differences.
- Official-test results were inspected during earlier development.
- This project does not implement the FSFP method described by Zhou et al.; that paper is cited as related work.
- No state-of-the-art performance claim is made.

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
