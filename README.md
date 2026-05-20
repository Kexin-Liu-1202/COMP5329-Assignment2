# COMP5329 Assignment 2

This repository contains our COMP5329 Assignment 2 implementation on CIFAR-10. The project studies how CNN architecture depth and width affect IID performance, OOD robustness, and representation complexity under a matched-parameter setting.

The final submission in this folder is based on the **50-epoch run** (`epoch50` / `e50`). Earlier pilot outputs have been removed, so the remaining results and checkpoints correspond to the final experiment setting.

## Project Summary

We compare three plain CNN variants with similar parameter budgets:

| Model | Type | Conv layers | Channels per stage | Parameters |
| --- | --- | ---: | --- | ---: |
| CNN-SW | shallow-wide | 6 | [64, 128, 256] | 1,148,874 |
| CNN-M | medium | 12 | [40, 80, 160] | 1,056,130 |
| CNN-DN | deep-narrow | 15 | [36, 72, 144] | 1,101,358 |

The training notebook runs all three models for 50 epochs on CIFAR-10, evaluates IID test accuracy, and then evaluates OOD robustness under four corruptions:

- Gaussian noise
- Blur
- Brightness shift
- Contrast shift

We also include representation analysis based on penultimate-layer features, including participation-ratio effective dimensionality, entropy-based effective rank, and spectrum plots.

## Final 50-Epoch Results

Best IID test accuracy:

- `CNN-SW`: 0.9151
- `CNN-M`: 0.9285
- `CNN-DN`: 0.9269

Mean OOD accuracy:

- `CNN-SW`: 0.6214
- `CNN-M`: 0.6570
- `CNN-DN`: 0.6600

Mean OOD drop:

- `CNN-SW`: 0.2937
- `CNN-M`: 0.2715
- `CNN-DN`: 0.2669

Under the final 50-epoch setting, the medium and deep-narrow models outperform the shallow-wide model on both IID accuracy and average OOD robustness.

## Repository Structure

```text
COMP5329-Assignment2.ipynb        Main notebook for training and evaluation
results_e50/                      Final 50-epoch training logs and evaluation outputs
checkpoints_e50/                  Best model checkpoints from the 50-epoch run
analysis/
  feature_complexity_analysis.py  Feature extraction and complexity analysis
  plot_auxiliary_e50.py           Extra plotting script for the 50-epoch analysis
  outputs_e50/                    Complexity-analysis CSV outputs
  plots_e50/                      Final plots used for analysis/reporting
  per_shift_analysis_e50.md       Written interpretation of the 50-epoch results
data/
  cifar-10-batches-py/            Extracted CIFAR-10 dataset
  cifar-10-python.tar.gz          Original CIFAR-10 archive
```

## Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

If you want to run the notebook interactively:

```bash
jupyter notebook
```

## Dataset Note

The CIFAR-10 dataset is not included in this GitHub repository.

We intentionally excluded the `data/` directory because:

- the raw dataset files are large and are not necessary for inspecting the code, trained models, and final results;
- GitHub is used here to share the implementation, final 50-epoch outputs, checkpoints, and analysis artifacts;
- CIFAR-10 can be downloaded separately from the official source if full reproduction from scratch is needed.

This repository already includes the final experiment outputs in `results_e50/`, the best checkpoints in `checkpoints_e50/`, and the analysis outputs in `analysis/`.

## How To Reproduce

### 1. Run the main notebook

Open and run:

```text
COMP5329-Assignment2.ipynb
```

The notebook is configured with:

- `EXPERIMENT_TAG = "e50"`
- `RESULTS_DIR = "results_e50"`
- `CHECKPOINT_DIR = "checkpoints_e50"`
- `num_epochs = 50`

Running the notebook will:

1. Train all three CNN variants on CIFAR-10.
2. Save epoch-level logs to `results_e50/training_logs.csv`.
3. Save best checkpoints to `checkpoints_e50/`.
4. Produce IID and OOD summary CSV files and core plots in `results_e50/`.

### 2. Run representation complexity analysis

The analysis script still has old generic defaults, so for the final `e50` outputs it should be run with explicit paths:

```bash
python analysis/feature_complexity_analysis.py \
  --data-root data \
  --checkpoint-dir checkpoints_e50 \
  --results-dir analysis/outputs_e50 \
  --plot-dir analysis/plots_e50 \
  --model-summary results_e50/model_summary.csv \
  --iid-results results_e50/final_iid_results.csv \
  --ood-summary results_e50/ood_summary.csv
```

This generates:

- `analysis/outputs_e50/complexity_metrics.csv`
- `analysis/outputs_e50/feature_spectra.csv`
- `analysis/outputs_e50/complexity_joined_summary.csv`
- plots such as `effective_dimensionality_pr.png` and `feature_spectrum_decay.png`

### 3. Generate auxiliary 50-epoch plots

```bash
python analysis/plot_auxiliary_e50.py
```

This writes extra OOD and representation figures into `analysis/plots_e50/`, including:

- `ood_drop_heatmap.png`
- `ood_accuracy_grouped_bar.png`
- `ood_drop_profile.png`
- `complexity_vs_per_shift_ood_drop.png`

## Main Output Files

Core training and evaluation files:

- `results_e50/model_summary.csv`
- `results_e50/training_logs.csv`
- `results_e50/final_iid_results.csv`
- `results_e50/ood_results.csv`
- `results_e50/ood_summary.csv`

Core plots:

- `results_e50/train_accuracy_curve.png`
- `results_e50/iid_test_accuracy_curve.png`
- `results_e50/ood_accuracy.png`
- `results_e50/ood_drop.png`
- `analysis/plots_e50/effective_dimensionality_pr.png`
- `analysis/plots_e50/feature_spectrum_decay.png`
- `analysis/plots_e50/complexity_vs_mean_ood_drop.png`
- `analysis/plots_e50/ood_drop_heatmap.png`

Written analysis:

- `analysis/per_shift_analysis_e50.md`

## Notes

- The dataset is expected to already exist in `data/cifar-10-batches-py/`.
- The notebook and scripts automatically use CUDA if available; otherwise they fall back to CPU.
- The final interpretation in this folder should use the `e50` outputs, not the earlier pilot setting.
