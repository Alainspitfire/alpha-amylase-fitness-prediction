# alpha-Amylase Fitness Prediction

Predicting alpha-amylase fitness using a mutation-aware Transformer ensemble, ESM-2 embeddings with Ridge regression, and a validation-selected hybrid model.

This project uses the FLIP2 alpha-amylase dataset. It includes training code, fitted model parameters, inference tools, and evaluation scripts.

## Models

- **Transformer:** An ensemble of three mutation-aware Transformers trained with seeds 42, 123, and 2026, using regression and pairwise ranking losses.
- **ESM–Ridge:** Mean-pooled embeddings from `facebook/esm2_t33_650M_UR50D`, expressed relative to a reference sequence, plus mutation count. Features are standardized using training data and passed to Ridge regression.
- **Hybrid:** A weighted average of the Transformer ensemble and ESM–Ridge predictions.

The final hybrid uses:

```text
Predicted fitness = 0.60 × Transformer + 0.40 × ESM–Ridge
```

The blending weight was selected using validation Spearman correlation.

## Evaluation

### Custom unseen-substitution test

The original official training and validation partitions were combined into a development pool of 3,004 sequences, then divided into:

| Partition | Sequences |
|---|---:|
| Training | 1,922 |
| Validation | 481 |
| Test | 601 |

The custom test contains sequences carrying at least one of 75 designated held-out substitutions. These substitutions are absent from the custom training and validation sequences.

| Model | Spearman ↑ | Pearson ↑ | RMSE ↓ | MAE ↓ | Pairwise accuracy ↑ |
|---|---:|---:|---:|---:|---:|
| Transformer | 0.62150 | 0.73340 | 0.03715 | **0.02520** | 88.51% |
| ESM–Ridge | 0.63981 | 0.67767 | 0.04424 | 0.03584 | 90.58% |
| Hybrid | **0.65449** | **0.75182** | **0.03611** | 0.02739 | **90.90%** |

### Official test partition

The same fitted models and validation-selected blending weight were evaluated on the 702-sequence official test partition without retraining.

| Model | Spearman ↑ | Pearson ↑ | RMSE ↓ | MAE ↓ | Pairwise accuracy ↑ |
|---|---:|---:|---:|---:|---:|
| Transformer | 0.65457 | 0.74310 | 0.03616 | **0.02447** | 89.05% |
| ESM–Ridge | 0.63187 | 0.64781 | 0.04388 | 0.03559 | 88.23% |
| Hybrid | **0.66380** | **0.74963** | **0.03560** | 0.02694 | **90.77%** |

This is secondary evaluation of models trained under the custom holdout protocol. It is not a model trained using the official benchmark split protocol.

Pairwise accuracy considers sequence pairs whose experimental fitness differs by at least 0.03. Prediction ties receive half credit.

The hybrid has the highest observed Spearman correlation on both tests. These point estimates alone do not establish statistically significant improvements.

## Installation

Run commands from the repository root.

Install dependencies with the compatibility pins used for the evaluation setup:

```bash
python -m pip install -r requirements.txt \
    "transformers==4.57.6" \
    "tokenizers==0.22.2" \
    "huggingface-hub==0.36.2" \
    "pandas==2.2.3" \
    "numpy==2.2.6"
```

A CUDA GPU is recommended for training and ESM inference. The scripts use CPU when CUDA is unavailable.

The dataset and pretrained ESM backbone download automatically when needed. Initial use requires internet access.

## Saved Model Files

Keep these folders directly beside the Python scripts:

```text
outputs_rank/
    rank_seed_42.pt
    rank_seed_123.pt
    rank_seed_2026.pt

outputs_esm_ridge/
    configuration.json
    ridge_head.npz

outputs_blend/
    ensemble_config.json
```

All three Transformer checkpoints are required for the final ensemble. Hybrid inference also requires the Ridge parameters, configurations, and pretrained ESM backbone.

Check that the required files exist:

```bash
python use_model.py setup --model hybrid
```

## Predict Fitness

Provide a FASTA file containing one 425-residue alpha-amylase sequence:

```bash
python use_model.py fitness --model hybrid --fasta sequence.fasta
```

For Transformer-only inference:

```bash
python use_model.py fitness --model transformer --fasta sequence.fasta
```

Sequences must contain standard amino-acid letters and match the reference length and residue alignment. The model does not support insertions or deletions.

Outputs are predicted fitness values on the dataset's target scale.

## Train the Models

To download the dataset, create the custom split, train both model components, select the blend, and evaluate:

```bash
python run_pipeline.py
```

Alternatively, run the stages individually:

```bash
python create_mutation_holdout.py
python train_rank.py
python train_esm_ridge.py --batch-size 2
python blend_predictions.py
python evaluate_hybrid.py
```

These commands write to the model-output folders and may overwrite existing artifacts. Preserve supplied results before retraining.

## Evaluate Saved Predictions and Models

Recalculate custom-test metrics from the saved prediction CSVs:

```bash
python evaluate_hybrid.py
```

Evaluate the existing fitted models on the official test partition:

```bash
python evaluate_official_test.py --root . --out-dir outputs_official_test
```

Keep model parameters and blending weights fixed when running official-test evaluation.

If `bootstrap_model_comparison.py` is included, calculate paired bootstrap intervals:

```bash
python bootstrap_model_comparison.py \
    --input outputs_official_test/official_test_predictions.csv \
    --out-dir outputs_official_test/bootstrap \
    --replicates 2000
```

The bootstrap samples sequences independently. Related variants may violate that assumption, so interpret intervals cautiously.

## Results and Data Layout

```text
mutation_holdout/             Split records and reference manifest
outputs_rank/                 Transformer checkpoints and predictions
outputs_esm_ridge/            Ridge parameters, configuration, and predictions
outputs_blend/                Hybrid configuration and predictions
outputs_hybrid_evaluation/    Custom-test metrics and figures
outputs_official_test/        Official-test predictions and metrics
```

The dataset downloader verifies the downloaded archive using an expected MD5 checksum.

## Limitations

- The model was developed for this alpha-amylase dataset and fixed sequence alignment.
- Performance on other proteins, insertions, deletions, or substantially different sequences has not been established.
- Predicted improvements require experimental validation.
- GPU hardware and software versions can cause small numerical differences.
- A higher overall ranking score does not imply improvement on every metric; the Transformer has lower MAE on both reported tests.

## Dataset Attribution

The dataset and experimental measurements come from **FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications**, by Didi et al. They are not original experimental contributions of this project.

- [FLIP2 paper](https://proceedings.mlr.press/v306/didi26a.html)
- [Dataset](https://doi.org/10.5281/zenodo.18433203)
- [Official benchmark](https://flip.protein.properties/)

## AI-Assisted Development

ChatGPT assisted with code development, debugging, evaluation workflows, and documentation. Reported performance values come from computational evaluations of the models.

## Reference

Dallago, Christian, et al. "FLIP: Benchmark tasks in fitness landscape inference for proteins." bioRxiv (2021): 2021-11.

Didi, Kieran, et al. "FLIP2: Expanding protein fitness landscape benchmarks for real-world machine learning applications." bioRxiv (2026): 2026-02.

Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.

Zhou, Ziyi, et al. "Enhancing efficiency of protein language models with minimal wet-lab data through few-shot learning." Nature Communications 15.1 (2024): 5566.
