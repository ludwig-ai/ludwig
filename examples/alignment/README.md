# LLM Alignment with DPO and KTO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/alignment/alignment_dpo.ipynb)

This example shows how to align a large language model with human preferences using Ludwig's
built-in preference learning trainers. Alignment training is typically applied after an initial
supervised fine-tuning (SFT) stage to improve response quality, reduce harmful outputs, and teach
the model to follow instructions more reliably.

## What is alignment?

Alignment refers to the process of shaping a model's behaviour to match human values and preferences.
The classic approach — Reinforcement Learning from Human Feedback (RLHF) — requires training a
separate reward model on human-ranked responses, then running a full RL loop (PPO) against it.
Modern preference learning methods like DPO bypass the reward model entirely, making alignment
cheaper and more stable to train.

## When to use each trainer

| Trainer | Data format                          | Use case                                                                     | Compute                                                                  |
| ------- | ------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `dpo`   | `prompt`, `chosen`, `rejected`       | Preference pairs from human feedback; most widely studied                    | Medium — requires forward passes through both policy and reference model |
| `kto`   | `prompt`, `response`, `label` (bool) | Single-label feedback (thumbs up/down); no paired responses needed           | Low — simpler loss than DPO                                              |
| `orpo`  | `prompt`, `chosen`, `rejected`       | Single-stage SFT + alignment; no separate reference model                    | Low — no reference model forward passes                                  |
| `grpo`  | `prompt`, custom reward function     | RL-style training with a group-normalised reward signal; used in DeepSeek-R1 | High — requires multiple rollouts per prompt                             |

Choose **DPO** when you have human-ranked response pairs and want the best-studied approach.
Choose **KTO** when collecting binary per-response feedback is easier than pairwise comparisons.
Choose **ORPO** when you want to skip the SFT stage and align in one shot.
Choose **GRPO** when you have a programmatic reward function (e.g. code execution, math verification).

## Prerequisites

- GPU with at least 40 GiB of VRAM (A100 recommended)
- [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
- Access approval to [Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

## Quick start

Install dependencies:

```bash
pip install "ludwig[llm]" datasets
```

Set your HuggingFace token:

```bash
export HUGGING_FACE_HUB_TOKEN="<your_token>"
```

Prepare the dataset:

```bash
python prepare_dataset.py
```

Run DPO training:

```bash
python train_dpo.py
# or with the CLI:
ludwig train --config config_dpo.yaml --dataset train.csv
```

Run KTO training:

```bash
ludwig train --config config_kto.yaml --dataset train_kto.csv
```

Run GRPO training (reuses the DPO preference-pair format):

```bash
python train_grpo.py
# or with the CLI:
ludwig train --config config_grpo.yaml --dataset preference_data.parquet
```

## GRPO specifics

GRPO (Group Relative Policy Optimization, Shao et al. 2024) is the alignment method used by
DeepSeek-R1. For each prompt it samples a group of `grpo_num_generations` completions, scores
them, normalises rewards within the group, and applies a PPO-style clipped objective —
without a separate critic model.

Ludwig's GRPO trainer consumes the same `prompt` / `chosen` / `rejected` columns as DPO, so
a programmatic reward function is implemented as a pre-processing step: score each candidate
completion in your dataset preparation pipeline, then emit the top-scoring completion as
`chosen` and the lowest as `rejected`. See `config_grpo.yaml` for the full list of knobs
(`grpo_beta` for the KL penalty, `grpo_epsilon` for PPO clipping,
`grpo_num_generations` for the group size).

## Files

| File                  | Description                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `prepare_dataset.py`  | Downloads Anthropic/hh-rlhf and converts it to Ludwig format       |
| `train_dpo.py`        | DPO training script using the Python API                           |
| `train_grpo.py`       | GRPO training script using the Python API                          |
| `config_dpo.yaml`     | Ludwig config for DPO                                              |
| `config_kto.yaml`     | Ludwig config for KTO                                              |
| `config_orpo.yaml`    | Ludwig config for ORPO                                             |
| `config_grpo.yaml`    | Ludwig config for GRPO                                             |
| `alignment_dpo.ipynb` | Colab-compatible notebook covering DPO, KTO evaluation, and upload |

## Upload to HuggingFace

After training, upload the aligned model:

```bash
ludwig upload hf_hub -r <your_org>/<model_name> -m results/experiment_run/model
```
