# HyperNetworkCombiner: Conditional Feature Processing

> **Note:** This example requires PR #4092 to be merged into Ludwig, or `pip install ludwig` >= 0.14.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/hypernetwork/hypernetwork.ipynb)

## What the HyperNetworkCombiner does differently

Most combiners — including the default `concat` combiner — treat all input features
symmetrically: they encode each feature independently and then merge the resulting
vectors (by concatenation, attention, or summation). The merged representation is
the same *kind* of computation regardless of what any individual feature says.

The `hypernetwork` combiner breaks this symmetry. One feature, called the
**conditioning feature**, is fed through a small *hyper-network* that generates the
weight matrices and biases of the fully-connected layers that process all other features.
In other words, the conditioning feature does not just *contribute* to the prediction —
it *rewrites the transformation* applied to every other feature before prediction happens.

This is based on **HyperFusion** (arXiv 2403.13319, 2024).

```
sensor_type  ──► HyperNetwork ──► generates weights W, b
                                        │
sensor_a ─────────────────────► FC(W, b) ──► combined
sensor_b ─────────────────────► FC(W, b) ──►  repr.
sensor_c ─────────────────────► FC(W, b) ──►
```

Contrast with concat:

```
sensor_type ──►  encoder  ──┐
sensor_a    ──►  encoder  ──┤
sensor_b    ──►  encoder  ──┼──► concat ──► FC ──► output
sensor_c    ──►  encoder  ──┘
```

With `concat`, the network learns *after* combining to react to different sensor types.
With `hypernetwork`, the combination itself is conditioned on sensor type.

## When to use it

Use the `hypernetwork` combiner when:

- One feature is a **context** or **mode** that fundamentally changes how other
  features should be interpreted (sensor type, device class, environment, language).
- The relationship between inputs and the target changes qualitatively across groups,
  not just quantitatively.
- You have enough training data to learn the per-context transformations (at minimum a
  few hundred samples per conditioning category).

Stick with `concat` when:

- All input features contribute on equal footing.
- The dataset is small (the hyper-network adds parameters).
- Interpretability of the encoding step is important and you want a fixed transformation.

## Files

| File                       | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `hypernetwork.ipynb`       | End-to-end walkthrough with synthetic sensor data                         |
| `config_concat.yaml`       | Baseline concat config                                                    |
| `config_hypernetwork.yaml` | HyperNetworkCombiner config                                               |
| `train_hypernetwork.py`    | Standalone script — generates data, trains both models, prints comparison |

## Quick start

```bash
pip install "ludwig>=0.14"
python train_hypernetwork.py
```
