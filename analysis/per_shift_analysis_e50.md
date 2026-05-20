# Per-shift OOD and Representation Analysis: 50-epoch Run

Source files:

- `results_e50/final_iid_results.csv`
- `results_e50/ood_results.csv`
- `analysis/outputs_e50/complexity_joined_summary.csv`

This note summarizes the 50-epoch run and should be treated as the stronger candidate for the final report. Compared with the 10-epoch pilot, all models are much better trained, and the relative robustness pattern becomes clearer.

## Summary

| Model | Conv layers | Effective dim. PR | Entropy rank | 95% energy comps. | Best IID acc. | Best epoch | Gen. gap | Mean OOD acc. | Mean OOD drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-SW | 6 | 7.868 | 13.580 | 22 | 0.9151 | 49 | 0.0671 | 0.6214 | 0.2937 |
| CNN-M | 12 | 8.230 | 10.372 | 9 | 0.9285 | 46 | 0.0663 | 0.6570 | 0.2715 |
| CNN-DN | 15 | 8.465 | 10.392 | 9 | 0.9269 | 50 | 0.0649 | 0.6600 | 0.2669 |

Main observation: after 50 epochs, effective dimensionality measured by participation ratio increases with depth: CNN-SW < CNN-M < CNN-DN. The deeper models also show higher mean OOD accuracy and smaller mean OOD drop. This is a stronger and more coherent trend than the 10-epoch pilot.

## Per-shift OOD Accuracy

| Model | Gaussian noise | Blur | Brightness shift | Contrast shift |
| --- | ---: | ---: | ---: | ---: |
| CNN-SW | 0.2111 | 0.4712 | 0.9024 | 0.9008 |
| CNN-M | 0.2671 | 0.5155 | 0.9222 | 0.9232 |
| CNN-DN | 0.2682 | 0.5281 | 0.9212 | 0.9226 |

## Per-shift OOD Drop

| Model | Gaussian noise | Blur | Brightness shift | Contrast shift |
| --- | ---: | ---: | ---: | ---: |
| CNN-SW | 0.7040 | 0.4439 | 0.0127 | 0.0143 |
| CNN-M | 0.6614 | 0.4130 | 0.0063 | 0.0053 |
| CNN-DN | 0.6587 | 0.3988 | 0.0057 | 0.0043 |

## Shift-specific Findings

### Gaussian Noise

CNN-DN and CNN-M are close under Gaussian noise, with CNN-DN slightly ahead. CNN-DN reaches 0.2682 OOD accuracy, CNN-M reaches 0.2671, and CNN-SW falls to 0.2111. The corresponding drops are 0.6587, 0.6614, and 0.7040.

This is the largest corruption-induced degradation across all shifts. Even after 50 epochs, Gaussian noise remains difficult for all models, but the deeper models are less affected than the shallow-wide model.

### Blur

CNN-DN performs best under blur with 0.5281 OOD accuracy and the smallest drop of 0.3988. CNN-M follows with 0.5155 accuracy and 0.4130 drop. CNN-SW is weakest, with 0.4712 accuracy and 0.4439 drop.

This is an important change from the 10-epoch pilot, where CNN-SW was best under blur. The 50-epoch result suggests that the earlier blur trend was likely affected by under-training, and that the deeper models become more robust once sufficiently optimized.

### Brightness Shift

Brightness shift is mild for all models. CNN-M is slightly best with 0.9222 accuracy, followed by CNN-DN at 0.9212 and CNN-SW at 0.9024. The drops are small for all models, especially CNN-M and CNN-DN.

Because this perturbation causes limited degradation, it is less useful for distinguishing robustness than Gaussian noise or blur. It mainly confirms that the models preserve performance under mild color/intensity shift.

### Contrast Shift

Contrast shift is also mild. CNN-M reaches the highest accuracy at 0.9232, CNN-DN is close at 0.9226, and CNN-SW is lower at 0.9008. CNN-DN has the smallest drop at 0.0043.

The result is consistent with the brightness shift pattern: medium and deep-narrow models retain performance better than the shallow-wide model.

## Representation Spectrum Interpretation

The participation-ratio effective dimensionality increases from 7.868 in CNN-SW to 8.230 in CNN-M and 8.465 in CNN-DN. This aligns with the robustness pattern in the 50-epoch run: models with higher participation ratio have higher mean OOD accuracy and lower mean OOD drop.

However, the spectrum concentration metrics show a more nuanced picture. CNN-SW has a much larger entropy-based effective rank and requires 22 components to explain 95% of feature energy. CNN-M and CNN-DN require only 9 components. This means CNN-SW has a longer spectral tail, while CNN-M and CNN-DN concentrate more energy in a smaller set of dominant directions.

For the report, participation ratio should be presented together with the spectrum decay plot. A single scalar complexity metric does not fully describe the representation geometry.

## Comparison with the 10-epoch Pilot

| Model | PR change | IID acc. change | Mean OOD acc. change | Mean OOD drop change |
| --- | ---: | ---: | ---: | ---: |
| CNN-SW | +1.053 | +0.0499 | +0.0043 | +0.0456 |
| CNN-M | +1.495 | +0.0634 | +0.0432 | +0.0202 |
| CNN-DN | +1.500 | +0.0725 | +0.0632 | +0.0093 |

The deeper models benefit more from longer training. CNN-DN improves the most in IID accuracy and mean OOD accuracy, while its mean OOD drop increases only slightly. CNN-SW improves in IID accuracy but has a larger increase in OOD drop, mainly because its IID accuracy improves while its Gaussian noise and blur robustness remain weaker.

## Claims Supported by the 50-epoch Run

The 50-epoch run supports the following cautious claims:

1. Under the current matched-parameter setup, CNN-M and CNN-DN outperform CNN-SW on IID accuracy and mean OOD robustness.
2. The deeper models are consistently better under the more severe corruptions, especially Gaussian noise and blur.
3. Participation-ratio effective dimensionality increases with depth and is associated with better mean OOD robustness in this run.
4. The relationship between representation complexity and robustness is suggestive, not conclusive, because the experiment contains only three model variants and one random seed.

## Recommended Use in the Paper

Use the 50-epoch results as the main results. Keep the 10-epoch run as a pilot or omit it from the main paper unless space permits an appendix note.

Recommended main figures:

1. IID test accuracy curve from `results_e50/iid_test_accuracy_curve.png`
2. OOD drop heatmap from `analysis/plots_e50/ood_drop_heatmap.png`
3. Grouped OOD accuracy bar chart from `analysis/plots_e50/ood_accuracy_grouped_bar.png`
4. Effective dimensionality bar plot from `analysis/plots_e50/effective_dimensionality_pr.png`
5. Feature spectrum decay from `analysis/plots_e50/feature_spectrum_decay.png`
6. Effective dimensionality vs mean OOD drop from `analysis/plots_e50/complexity_vs_mean_ood_drop.png`

Recommended auxiliary figures:

1. Robustness profile from `analysis/plots_e50/ood_drop_profile.png`
2. Effective dimensionality vs per-shift OOD drop from `analysis/plots_e50/complexity_vs_per_shift_ood_drop.png`

Main limitation to state: the final experiment uses three architectures and one seed, so trends should be interpreted as controlled empirical evidence rather than statistically definitive conclusions.
