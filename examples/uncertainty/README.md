# Uncertainty Quantification: MC Dropout and Temperature Scaling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/uncertainty/uncertainty.ipynb)

## Overview

This example demonstrates two practical techniques for quantifying and reducing model uncertainty in Ludwig:

- **Temperature Scaling Calibration**: Post-hoc calibration that adjusts overconfident predicted probabilities to better match empirical frequencies. Based on [Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599).
- **MC Dropout**: Monte Carlo Dropout runs multiple stochastic forward passes at inference time to produce per-sample uncertainty estimates. Based on [Gal & Ghahramani, ICML 2016](https://arxiv.org/abs/1506.02142).

Both techniques are applied to a binary wine quality classifier (UCI Wine Quality dataset) to illustrate when each method is appropriate and how to configure them in Ludwig.

## What You Will Learn

1. Why deep learning models are often overconfident and why calibration matters
1. How to enable temperature scaling in a Ludwig config (one line change)
1. How to compute Expected Calibration Error (ECE) and plot reliability diagrams
1. How to enable MC Dropout for per-sample uncertainty estimates
1. How to interpret the `uncertainty` output alongside predictions

## Prerequisites

- Python 3.9+
- Ludwig installed (`pip install ludwig`)
- Internet access to download the UCI Wine Quality dataset (~80 KB)

Optional (for GPU training):

```
pip install ludwig[gpu]
```

## Quick Start

### Run the notebook

Click the Colab badge above, or open `uncertainty.ipynb` locally with Jupyter.

### Run the standalone script

```bash
pip install ludwig
python train.py
```

This will:

1. Download the red wine quality dataset from UCI
1. Train three models: baseline, temperature-scaled, and MC Dropout
1. Print Expected Calibration Error for each model
1. Save reliability diagram plots to `./visualizations/`

## Files

| File                     | Description                               |
| ------------------------ | ----------------------------------------- |
| `uncertainty.ipynb`      | Interactive Colab notebook walkthrough    |
| `train.py`               | Standalone training and evaluation script |
| `config_baseline.yaml`   | Baseline config — no calibration          |
| `config_calibrated.yaml` | Config with temperature scaling enabled   |
| `config_mc_dropout.yaml` | Config with MC Dropout enabled            |

## Dataset

UCI Wine Quality (red wine), 1,599 samples, 11 physicochemical features.
Binary target: quality score >= 7 is "good" (positive class), otherwise "bad".
Class imbalance (~14% positive) makes calibration especially important.

## Key Results

| Model               | ECE   | Notes                                                |
| ------------------- | ----- | ---------------------------------------------------- |
| Baseline            | ~0.12 | Overconfident — probabilities cluster near 0/1       |
| Temperature Scaling | ~0.04 | Better calibrated, same accuracy                     |
| MC Dropout          | —     | Outputs per-sample uncertainty alongside predictions |

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*.
