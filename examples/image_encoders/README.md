# Pretrained Image Encoders: CLIP, DINOv2, and SigLIP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/image_encoders/image_encoders.ipynb)

## Overview

Pretrained image encoders are neural networks trained on large datasets (ImageNet-21k, LAION-5B, or proprietary corpora) whose learned weights can be directly transferred to new tasks. Instead of training a convolutional network from scratch on your small dataset, you can use a pretrained encoder as a frozen feature extractor and only train a lightweight classification head on top—this is called **linear probing**.

### Why pretrained encoders matter for few-shot learning

When you have limited labeled data (e.g., 5–100 examples per class), training from scratch typically leads to overfitting. Pretrained encoders solve this by:

- Providing rich, general-purpose visual features learned from millions of images
- Allowing the model to converge in far fewer epochs
- Requiring only a small head to be trained, which needs very little data

Ludwig supports three HuggingFace-backed pretrained image encoders alongside the traditional `stacked_cnn` approach.

## Encoder comparison

| Encoder       | Pretrained | Trainable by default | Best for                                                                    |
| ------------- | ---------- | -------------------- | --------------------------------------------------------------------------- |
| `stacked_cnn` | No         | Yes                  | Full control, small images, custom architectures                            |
| `dinov2`      | Yes        | Yes                  | General image classification, dense prediction, linear probing              |
| `clip`        | Yes        | Yes                  | Image-text tasks, zero-shot classification, multimodal fusion               |
| `siglip`      | Yes        | Yes                  | CLIP-like tasks with better scaling, Google's improved contrastive training |

All three pretrained encoders (`dinov2`, `clip`, `siglip`) support:

- `use_pretrained: true` — load weights from HuggingFace Hub
- `trainable: false` — freeze the encoder for fast linear probing
- `trainable: true` — fine-tune the full encoder end-to-end

## Quick start

Install Ludwig with vision support:

```bash
pip install ludwig[vision]
```

Train with a pretrained DINOv2 encoder (linear probe — fast, works well with limited data):

```bash
ludwig train \
  --config examples/image_encoders/config_dinov2_linear_probe.yaml \
  --dataset my_images.csv
```

Your CSV needs two columns: `image_path` (absolute or relative paths to image files) and `label` (the class name).

## Available configs

| Config file                       | Description                                   |
| --------------------------------- | --------------------------------------------- |
| `config_stacked_cnn.yaml`         | CNN trained from scratch (20 epochs)          |
| `config_dinov2_linear_probe.yaml` | DINOv2 frozen backbone, head only (10 epochs) |
| `config_dinov2_finetuned.yaml`    | DINOv2 full fine-tune (5 epochs, lower LR)    |
| `config_clip.yaml`                | CLIP frozen backbone (10 epochs)              |
| `config_siglip.yaml`              | SigLIP frozen backbone (10 epochs)            |

## Running all configs and comparing results

```bash
python examples/image_encoders/compare_encoders.py --dataset my_images.csv
```

## Full walkthrough

See the [notebook](image_encoders.ipynb) for a complete step-by-step example using the `beans` plant disease dataset (3 classes, ~1000 images) from HuggingFace Datasets, including a few-shot experiment with only 15 training examples.

## Hardware requirements

- `stacked_cnn`: CPU or GPU
- `dinov2` (linear probe): GPU recommended, runs on CPU for small datasets
- `dinov2` (fine-tune), `clip`, `siglip`: GPU required (T4 or better)

The linear probe is especially well-suited for Google Colab free tier (T4 GPU).
