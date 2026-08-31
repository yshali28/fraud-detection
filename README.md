# Adaptive Fraud Detection via Continual Learning

Credit card fraud patterns drift over time, but most fraud classifiers are trained once and left static, so their accuracy decays as fraudsters change tactics. Naively retraining on every new batch of transactions risks catastrophic forgetting (the model overwrites what it learned from older fraud patterns) and, on a stream that is >99.8% legitimate transactions, is expensive to do continuously. This project simulates a streaming, multi-era fraud environment (the Kaggle Credit Card Fraud dataset split into 4 chronological eras) and implements a continual learning system that retrains only when it detects meaningful drift, using a generative replay buffer to preserve knowledge of past fraud patterns without storing raw historical data.

## Approach

- **Class-conditional VAE-based generative replay** — a separate Variational Autoencoder is trained per era for the fraud class and the non-fraud class (`src/vae.py`). Instead of retaining raw historical transactions, the system samples synthetic fraud/non-fraud examples from these VAEs when retraining, avoiding both a growing raw-data buffer and the class imbalance problem of naive replay.
- **ADWIN-based drift detection** — a dual-signal drift detector (`src/drift_detector.py`) built on `river`'s ADWIN. One ADWIN instance tracks overall prediction error; a second tracks error only on "fraud-suspicious" predictions (probability > 0.3), since on a stream that's >99.8% non-fraud, overall error rate barely moves even when the model misses most fraud — the fraud-suspicious channel makes drift ~100x more visible.
- **Drift-triggered retraining** — the continual learner (`src/continual_learner.py`) only fully retrains (with VAE-sampled replay from all prior eras) when drift is detected; otherwise it does a light fine-tune on the current era. This is what drives the compute savings below.
- **Baselines for comparison** (`src/baselines.py`): no-replay retraining, raw-data replay, and VAE-always-on (retrain with replay every era regardless of drift).

## Results

Evaluated across 4 chronological eras of the Kaggle Credit Card Fraud dataset:

| Metric | Value |
|---|---|
| Best per-era F1 (drift-triggered system) | **0.8561** |
| Compute reduction vs. always-retrain baseline | **67%** |
| Backward transfer | **-0.0721** |

Backward transfer measures how much performance on earlier eras degrades as the model learns new ones (0 = no forgetting, negative = some forgetting) — see `outputs/plots/backward_transfer.png` and `outputs/results/results.csv` for the full per-era breakdown against all baselines.

## Tech stack

Python, PyTorch (VAE, classifier), [river](https://riverml.xyz/) (ADWIN drift detection), scikit-learn, pandas/NumPy, matplotlib/seaborn.

## Project structure

```
.
├── main.py                  # Entry point: runs baselines + continual learning system, saves plots/results
├── src/
│   ├── preprocess.py        # Loads and splits data into chronological eras
│   ├── vae.py                # Class-conditional VAE model + training + sampling
│   ├── drift_detector.py     # ADWIN-based dual-signal drift detector
│   ├── continual_learner.py  # Drift-triggered retraining loop with VAE replay
│   ├── baselines.py          # No-replay / raw-replay / VAE-always-on baselines
│   ├── classifier.py         # Fraud classifier model + training + threshold tuning
│   └── evaluate.py           # Metrics, plotting, results export
├── notebooks/
│   └── exploration.ipynb    # Exploratory data analysis on the raw dataset
├── data/
│   └── raw/                 # Place creditcard.csv here (not included — see below)
└── outputs/
    ├── plots/                # Generated figures (F1/AUC per era, backward transfer, t-SNE, retraining events)
    ├── results/              # results.csv — full per-era metrics for all systems
    └── models/               # (empty — models are not checkpointed to disk by default)
```

## Setup

```bash
pip install -r requirements.txt
```

**Dataset**: this repo does not include the raw data. Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`.

## Running

```bash
python main.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--data_path` | `data/raw/creditcard.csv` | Path to the dataset CSV |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--seed` | `42` | Random seed |
| `--skip_baselines` | off | Skip the 3 baseline systems and only run the continual learning system |
| `--output_dir` | `outputs` | Where plots/results are written |

Results are written to `outputs/results/results.csv` and plots to `outputs/plots/`.
