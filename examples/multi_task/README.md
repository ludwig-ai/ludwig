# Multi-Task Learning with Nash-MTL Loss Balancing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/multi_task/multi_task.ipynb)

> **Note:** Nash-MTL requires PR #4092 (`future-capabilities` branch) and is not yet available in the main Ludwig release. The FAMO and uncertainty weighting methods shown here are available now.

## Overview

This example demonstrates multi-task learning with Ludwig: training a single model to predict multiple outputs simultaneously, and using **loss balancing** to prevent one task from dominating training.

The dataset is the [UCI Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). We predict two outputs at once:

- `quality_score` — the raw 0–10 quality score (regression)
- `quality_binary` — whether the wine is good (quality ≥ 7, binary classification)

These two tasks have different loss magnitudes. Without balancing, the regression loss typically dominates and the classifier under-trains.

## What You Will Learn

1. How to define multiple output features in a Ludwig config
1. Why loss magnitudes differ between regression and classification tasks
1. How FAMO and uncertainty weighting improve multi-task training (available now)
1. What Nash-MTL does and how it compares to heuristic methods (requires PR #4092)
1. How to read a comparison table and choose the right balancing strategy

## Loss Balancing Methods Compared

| Method          | Status            | When to use                                    |
| --------------- | ----------------- | ---------------------------------------------- |
| `none`          | Available         | Baseline; tasks have similar loss scales       |
| `log_transform` | Available         | Quick improvement with no hyperparameters      |
| `uncertainty`   | Available         | Tasks have stable, learnable scale differences |
| `famo`          | Available         | General purpose; good default choice           |
| `gradnorm`      | Available         | Gradient-level balancing; more expensive       |
| `nash_mtl`      | Requires PR #4092 | Most principled; best when tasks conflict      |

## Quick Start

```bash
pip install ludwig

# Baseline
ludwig train --config config_no_balancing.yaml --dataset wine_quality_dual.csv

# FAMO (available now)
ludwig train --config config_famo.yaml --dataset wine_quality_dual.csv

# Uncertainty weighting (available now)
ludwig train --config config_uncertainty.yaml --dataset wine_quality_dual.csv

# Nash-MTL (requires PR #4092)
ludwig train --config config_nash_mtl.yaml --dataset wine_quality_dual.csv
```

Or run the full comparison script:

```bash
python train_multi_task.py
```

## Files

| File                       | Description                                |
| -------------------------- | ------------------------------------------ |
| `multi_task.ipynb`         | Interactive notebook with full walkthrough |
| `train_multi_task.py`      | Standalone Python script                   |
| `config_no_balancing.yaml` | Baseline config — no loss balancing        |
| `config_famo.yaml`         | FAMO balancing (available now)             |
| `config_uncertainty.yaml`  | Uncertainty weighting (available now)      |
| `config_nash_mtl.yaml`     | Nash-MTL balancing (requires PR #4092)     |

## Prerequisites

- Python 3.9+
- Ludwig installed (`pip install ludwig`)
- Internet access to download the UCI Wine Quality dataset (~80 KB)

Optional: GPU for faster training (not required).

## Background

### Multi-Task Learning

Multi-task learning trains a shared model to predict several outputs simultaneously. The shared representation encourages the model to learn features useful across tasks, often improving generalisation compared to separate single-task models — especially when training data is limited.

### The Loss Balancing Problem

When tasks have different loss scales (e.g., MSE for regression vs. cross-entropy for binary classification), their gradients have different magnitudes. During backpropagation, the task with larger gradients dominates parameter updates and the other task effectively under-trains.

Loss balancing methods assign adaptive weights to each task's loss so that all tasks contribute proportionately to the total gradient.

### Nash-MTL

Nash-MTL (Navon et al., ICML 2022) frames loss balancing as a Nash bargaining game. Rather than using heuristic rules or hand-tuned weights, it finds the unique solution where no task can improve its loss without worsening another task's loss. This makes it the most principled approach, particularly valuable when tasks genuinely conflict.

See [Navon et al., 2022](https://arxiv.org/abs/2202.01017) for the theoretical grounding.
