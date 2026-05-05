# Advanced PEFT Adapters in Ludwig

This directory contains examples demonstrating Ludwig's extended PEFT (Parameter-Efficient Fine-Tuning)
adapter support, including:

- **PiSSA / EVA / CorDA / LoftQ** — advanced LoRA initializers
- **rsLoRA** — rank-stabilized LoRA scaling
- **TinyLoRA** — extreme low-rank fine-tuning (LoRA-XS variant)
- **C3A** — contextual/conditional/compositional adapters
- **OFT / HRA** — orthogonal fine-tuning methods
- **WaveFT** — wavelet-domain fine-tuning
- **LN-Tuning** — layer normalization only
- **VBLoRA** — vector bank LoRA

## Files

| File                  | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `pissa_lora.yaml`     | PiSSA initialization (faster convergence than standard LoRA) |
| `eva_lora.yaml`       | EVA initialization (data-driven, SOTA performance)           |
| `corda_lora.yaml`     | CorDA initialization (combines PiSSA + context signals)      |
| `loftq_lora.yaml`     | LoftQ (quantization-aware LoRA init)                         |
| `rslora_dora.yaml`    | rsLoRA + DoRA combination                                    |
| `tinylora_llm.yaml`   | TinyLoRA for LLM fine-tuning on minimal hardware             |
| `c3a_llm.yaml`        | C3A adapter for multi-task scenarios                         |
| `oft_llm.yaml`        | OFT adapter (orthogonal, preserves pretrained knowledge)     |
| `hra_llm.yaml`        | HRA adapter (Householder reflections)                        |
| `waveft_llm.yaml`     | WaveFT adapter (frequency-domain updates)                    |
| `ln_tuning_llm.yaml`  | LN-Tuning (ultra-lightweight: only LayerNorm weights)        |
| `vblora_llm.yaml`     | VBLoRA (shared vector bank for extreme compression)          |
| `compare_adapters.py` | Script comparing adapters by parameter count                 |
| `train_example.py`    | Full training example with adapter selection                 |

## Quick Start

```bash
# Train with PiSSA (recommended for most tasks — faster convergence)
ludwig train --config pissa_lora.yaml --dataset ludwig://imdb

# Ultra-low memory: TinyLoRA
ludwig train --config tinylora_llm.yaml --dataset ludwig://imdb

# Orthogonal fine-tuning (preserves pretrained knowledge)
ludwig train --config oft_llm.yaml --dataset ludwig://imdb
```

## Adapter Selection Guide

| Hardware constraint | Recommended adapter      | Params (7B model) |
| ------------------- | ------------------------ | ----------------- |
| 80 GB GPU           | `lora` r=16 + PiSSA init | ~100M             |
| 24 GB GPU           | `lora` r=8 + rsLoRA      | ~50M              |
| 16 GB GPU           | `tinylora` r=2           | ~1M               |
| 8 GB GPU            | `ln_tuning`              | ~0.1M             |
| Edge / CPU          | `tinylora` r=2, u=13     | \<100K            |
