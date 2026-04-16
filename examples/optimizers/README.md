# Optimizer Guide

Ludwig 0.14 added five optimizers on top of the PyTorch family: **RAdam**, **Adafactor**,
**Schedule-Free AdamW**, **Muon**, and **SOAP**. This tutorial walks through when to reach
for each one, how their config surfaces differ, and a side-by-side training-curve comparison
on the Wine Quality regression benchmark.

## Quick picks

| Optimizer           | `type`                | Best for                                                | Extra dependency |
| ------------------- | --------------------- | ------------------------------------------------------- | ---------------- |
| AdamW               | `adamw`               | Default baseline                                        | —                |
| RAdam               | `radam`               | Drop-in AdamW replacement, no warmup needed             | —                |
| Adafactor           | `adafactor`           | Memory-efficient LLM fine-tuning                        | `transformers`   |
| Schedule-Free AdamW | `schedule_free_adamw` | Cosine-decay AdamW without a scheduler                  | `schedulefree`   |
| Muon                | `muon`                | Stable large-scale pretraining                          | —                |
| SOAP                | `soap`                | Shampoo-style preconditioner (research-heavy workloads) | `soap-pytorch`   |

## Running the comparison

The `optimizer_comparison.py` script trains the same three-layer tabular model on the
Wine Quality dataset with each optimizer in turn and plots loss / RMSE curves side by side.

```bash
pip install 'ludwig[viz]'
python optimizer_comparison.py
```

It writes `optimizer_comparison.png` and `optimizer_comparison.csv` to the current directory.
The run is CPU-friendly (no GPU required) and completes in a few minutes.

## Key config differences to watch for

### RAdam

Treat RAdam as a drop-in AdamW replacement. Its on-the-fly variance rectification makes a
linear learning-rate warmup unnecessary in most cases.

```yaml
trainer:
  optimizer:
    type: radam
    betas: [0.9, 0.999]
    weight_decay: 0.01
```

### Adafactor

`relative_step: true` (the default) makes Adafactor manage its **own** learning rate
schedule from per-parameter statistics. **Do not also attach a `learning_rate_scheduler`**
— the two will fight and training will diverge. Leave `trainer.learning_rate` unset too:

```yaml
trainer:
  optimizer:
    type: adafactor
    relative_step: true
    scale_parameter: true
    warmup_init: false
```

Set `relative_step: false` only if you want the externally-configured `trainer.learning_rate`
plus an external scheduler.

### Schedule-Free AdamW

Matches cosine-decay AdamW without any scheduler. Ludwig handles the required
`optimizer.train()` / `optimizer.eval()` toggles around training and evaluation
automatically.

```yaml
trainer:
  optimizer:
    type: schedule_free_adamw
    betas: [0.9, 0.999]
    weight_decay: 0.01
    warmup_steps: 1000
```

### Muon

Muon uses momentum plus a Newton–Schulz orthogonalization of each weight-matrix update.
Its default learning rate (`0.02`) is **roughly 20× larger** than the typical AdamW starting
point. Always retune the learning rate when switching from Adam.

```yaml
trainer:
  learning_rate: 0.02     # Muon's default sweet spot
  optimizer:
    type: muon
    momentum: 0.95
    nesterov: true
```

### SOAP

Shampoo-as-Adam-Preconditioner. High memory use (one Kronecker factor per matrix plus the
usual Adam buffers), so it makes sense for research-scale setups where the preconditioner
cost is amortized by long training runs.

```yaml
trainer:
  optimizer:
    type: soap
    betas: [0.95, 0.95]
    weight_decay: 0.01
```

## When to pick which

```
Do you want a one-line swap from AdamW with no surprises?
  └─ radam

Are you fine-tuning a large model and short on GPU memory?
  └─ adafactor (with relative_step: true, no external scheduler)

Do you want cosine-decay behaviour without configuring a scheduler?
  └─ schedule_free_adamw

Are you training from scratch and can retune the learning rate?
  └─ muon (lr≈0.02) — stable, well-conditioned updates

Are you in a research setting and happy to trade memory for preconditioning?
  └─ soap
```

Everything else: `adamw` remains a safe default.

## References

- RAdam — Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond", ICLR 2020.
  <https://arxiv.org/abs/1908.03265>
- Adafactor — Shazeer & Stern, "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost",
  ICML 2018. <https://arxiv.org/abs/1804.04235>
- Schedule-Free AdamW — Defazio et al., "The Road Less Scheduled", 2024.
  <https://arxiv.org/abs/2405.15682>
- Muon — Jordan et al., "Muon: An optimizer for hidden layers in neural networks", 2024.
  <https://kellerjordan.github.io/posts/muon/>
- SOAP — Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam", 2024.
  <https://arxiv.org/abs/2409.11321>
