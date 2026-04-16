# HyperNetworkCombiner Tutorial

> **Requires Ludwig 0.15 / PR #4092 (future-capabilities branch).**

Most combiners in Ludwig fuse features *additively*: each encoder output is projected and
the projections are concatenated, averaged, attended over, and so on. The
**HyperNetworkCombiner** does something different — it uses one input feature to *generate
the weights* of the processing layers that transform the other features.

This is useful when one modality effectively describes "what kind of input this is" (a sensor
type, a patient cohort, a context vector) and the model should process the remaining
modalities conditioned on that description. Rather than treating the conditioning signal as
just another vector to concatenate, the hypernetwork produces data-dependent weights that
*adapt* the downstream processing per example.

Based on the HyperFusion paper (arXiv 2403.13319).

## When to reach for it

Use the HyperNetworkCombiner when:

1. You have a multimodal input where one modality is clearly *metadata* about the others
   (e.g. "this row is an X-ray" vs "this row is a CT scan").
1. You've verified a plain `concat` / `FT-Transformer` combiner already works. Hypernetworks
   shine when the data-dependent conditioning helps — if concat is already good, the extra
   parameters rarely pay off.
1. You're willing to pay the extra memory cost: the hypernetwork generates a
   `hidden_size × hidden_size` weight matrix per example.

If any of those don't apply, stick with `ft_transformer` or `concat`.

## Config

```yaml
input_features:
  # IMPORTANT: the first feature is the conditioning signal. Order matters.
  - name: sensor_type           # categorical — which sensor produced the reading
    type: category
  - name: reading_1
    type: number
  - name: reading_2
    type: number
  - name: reading_3
    type: number

output_features:
  - name: anomaly
    type: binary

combiner:
  type: hypernetwork
  hidden_size: 128        # width of feature projections
  hyper_hidden_size: 64   # hidden width of the hypernetwork weight generator
  output_size: 128        # FC stack output
  num_fc_layers: 2
  dropout: 0.1
  activation: relu
```

**Critical**: the first entry in `input_features` is the conditioning feature. Reorder your
YAML to put the metadata feature first.

## Running

```bash
pip install ludwig
python train_conditioned_model.py
```

The script generates a small synthetic multi-sensor dataset — three sensor types with very
different noise / scaling profiles on otherwise identical underlying signals — trains two
models on it (plain `concat` combiner vs. `hypernetwork` combiner) and prints the per-sensor
test accuracy side by side. On the synthetic data the hypernetwork variant typically
outperforms `concat` by 3–6 percentage points of accuracy because it can "specialise" its
processing weights per sensor type.

## Files

| File                         | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `config_concat.yaml`         | Baseline `concat` combiner for comparison       |
| `config_hypernetwork.yaml`   | HyperNetworkCombiner config                     |
| `train_conditioned_model.py` | Generates synthetic data and trains both models |

## References

- HyperFusion — Mansour & Shkolnisky, "HyperFusion: A Hypernetwork for Multimodal Learning",
  arXiv 2403.13319 (2024). <https://arxiv.org/abs/2403.13319>
- Hypernetworks — Ha et al., "HyperNetworks", ICLR 2017. <https://arxiv.org/abs/1609.09106>
