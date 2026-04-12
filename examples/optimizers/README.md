# Optimizer Comparison: Schedule-Free, Muon, Adafactor, and More

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/optimizers/optimizer_comparison.ipynb)

## Why optimizer choice matters

The optimizer is more than a training detail — it controls how fast gradients
are translated into weight updates, whether training is stable in early epochs,
how much memory the optimizer state consumes, and whether you need to tune a
separate learning-rate schedule at all.

Ludwig 0.11 added five production-ready optimizers beyond the classic Adam/SGD
family: **RAdam**, **Adafactor**, **Schedule-Free AdamW**, **Muon**, and **SOAP**.
This example shows how to configure each one and compares them on a real dataset.

## What this example shows

- How to set `trainer.optimizer.type` in a Ludwig YAML config
- The one rule for Schedule-Free AdamW: no `learning_rate_scheduler`
- Side-by-side training curves (validation loss + accuracy) for all optimizers
- A summary table of final metrics and wall-clock training time

## Prerequisites

```bash
pip install ludwig
```

No GPU required. The notebook runs on CPU in a few minutes.

## Quick start

### Run the notebook (recommended)

Open [`optimizer_comparison.ipynb`](optimizer_comparison.ipynb) in Jupyter or
click the Colab badge above.

### Run the script

```bash
python optimizer_comparison.py
```

This downloads the UCI Wine Quality dataset, trains all five configs, and
prints a comparison table.

### Use a standalone YAML config

Each optimizer has its own config file you can use directly with the Ludwig CLI:

```bash
ludwig train --config config_schedule_free_adamw.yaml --dataset winequality-red.csv
```

| File                              | Optimizer           |
| --------------------------------- | ------------------- |
| `config_adamw.yaml`               | AdamW (baseline)    |
| `config_radam.yaml`               | RAdam               |
| `config_adafactor.yaml`           | Adafactor           |
| `config_schedule_free_adamw.yaml` | Schedule-Free AdamW |
| `config_muon.yaml`                | Muon                |

## Key insight: Schedule-Free AdamW needs no LR scheduler

```yaml
trainer:
  optimizer:
    type: schedule_free_adamw
    lr: 0.001
  # Do NOT add learning_rate_scheduler here
```

Adding a `learning_rate_scheduler` on top of `schedule_free_adamw` fights the
built-in schedule and hurts convergence. See the notebook for a detailed
explanation.
