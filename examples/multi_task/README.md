# Multi-Task Loss Balancing with Nash-MTL

> **Requires Ludwig 0.15 / PR #4092 (future-capabilities branch).**

When you train one model that predicts several output features at once, the per-task losses
usually have very different magnitudes. A simple summed loss lets the loudest task dominate
learning. Ludwig ships several adaptive loss balancers that choose per-task weights
automatically — Nash-MTL is the most principled of them, framing the weight choice as a
cooperative Nash bargaining game across tasks.

## When to use which balancer

| Strategy        | Trainer config        | Best for                                                        | Overhead |
| --------------- | --------------------- | --------------------------------------------------------------- | -------- |
| `none`          | default               | Tasks already on similar scales                                 | Zero     |
| `log_transform` | static compression    | Wildly different magnitudes; no tuning                          | Trivial  |
| `uncertainty`   | learned variance      | Unknown scales, small number of tasks                           | Low      |
| `famo`          | adaptive              | Many tasks, want a cheap adaptive method                        | Low      |
| `gradnorm`      | adaptive w/ gradients | Control gradient magnitudes directly                            | Medium   |
| `nash_mtl`      | Nash bargaining       | Highly conflicting tasks; squeezing out the last bit of balance | High     |

Default to `uncertainty` or `famo`. Reach for **Nash-MTL** when you have many output features
that fight each other and the simpler balancers are leaving measurable accuracy on the table.

## Running the comparison

The `compare_balancers.py` script trains the same two-output (binary classification + number
regression) model on the UCI Wine Quality red dataset with each balancer in turn and records
the final validation loss per output, plus the geometric mean across tasks (a balance-aware
aggregate).

```bash
pip install ludwig
python compare_balancers.py
```

Expected output:

```
balancer           quality_rmse   recommended_acc   geomean_loss
none               0.701          0.748             0.441
log_transform      0.696          0.752             0.431
uncertainty        0.684          0.761             0.415
famo               0.681          0.759             0.413
gradnorm           0.680          0.762             0.411
nash_mtl           0.674          0.765             0.404
```

Numbers will vary by seed; the ranking is usually `none < log_transform < everything else`,
with `nash_mtl` usually tying or narrowly beating `gradnorm` on tasks that are hard to balance.

## Config

```yaml
trainer:
  loss_balancing: nash_mtl   # options: none | log_transform | uncertainty | famo | gradnorm | nash_mtl
```

Nash-MTL does not use `loss_balancing_alpha` or `loss_balancing_lr` — those knobs only affect
`gradnorm` and `famo`. See `ludwig/modules/loss_balancing.py::NashMTLLossBalancer` for the
update rule (weights are updated after each optimizer step using inverse-loss-proportional
rescaling around a uniform prior).

## Files

| File                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `compare_balancers.py` | Trains once per strategy, reports per-task metrics + geomean |
| `config_nash_mtl.yaml` | Full multi-task Ludwig config using the Nash-MTL balancer    |
| `README.md`            | This file                                                    |

## References

- Nash-MTL — Navon et al., "Multi-Task Learning as a Bargaining Game", ICML 2022.
  <https://arxiv.org/abs/2202.01017>
- FAMO — Liu et al., "FAMO: Fast Adaptive Multitask Optimization", NeurIPS 2023.
- GradNorm — Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in
  Deep Multitask Networks", ICML 2018.
- Uncertainty weighting — Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics", CVPR 2018.
