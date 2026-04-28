# Mamba-2 and Jamba Sequence Encoders

Ludwig ships two SSM-based sequence encoders for text, audio, and timeseries features:

| Encoder  | Architecture                                 | Best for                                   |
| -------- | -------------------------------------------- | ------------------------------------------ |
| `mamba2` | Pure SSM stack (Dao & Gu, 2024)              | Long sequences, compute-budget constrained |
| `jamba`  | Hybrid SSM + Attention (Lieber et al., 2024) | Balance between speed and global context   |

Both run in **linear time** in sequence length, making them competitive with Transformers on
documents longer than ~1 K tokens.

## Files

| file                              | description                                   |
| --------------------------------- | --------------------------------------------- |
| `mamba2_text_classification.yaml` | News article classification with pure Mamba-2 |
| `jamba_sequence.yaml`             | Sentiment analysis with hybrid Jamba encoder  |

## Quick start

```bash
# Download a sample dataset (AG News)
python -c "
from datasets import load_dataset
ds = load_dataset('ag_news', split='train[:5000]')
ds.to_csv('ag_news_train.csv', index=False)
ds = load_dataset('ag_news', split='test[:1000]')
ds.to_csv('ag_news_test.csv', index=False)
"

# Rename columns to match the config
python -c "
import pandas as pd
for split in ['train', 'test']:
    df = pd.read_csv(f'ag_news_{split}.csv')
    df = df.rename(columns={'text': 'article', 'label': 'category'})
    df.to_csv(f'ag_news_{split}.csv', index=False)
"

# Train with Mamba-2
ludwig train \
  --config mamba2_text_classification.yaml \
  --dataset ag_news_train.csv

# Train with Jamba
ludwig train \
  --config jamba_sequence.yaml \
  --dataset ag_news_train.csv  # rename 'article' column to 'review_text' first
```

## Key hyperparameters

| parameter           | default                | description                                          |
| ------------------- | ---------------------- | ---------------------------------------------------- |
| `d_model`           | 256                    | Hidden dimension (must be divisible by `num_heads`)  |
| `n_layers`          | 4 (mamba2) / 8 (jamba) | Number of blocks                                     |
| `num_heads`         | 8                      | Heads for the multi-head SSD decay                   |
| `d_conv`            | 4                      | Depthwise convolution kernel size                    |
| `expand_factor`     | 2                      | Inner expansion: `d_inner = d_model × expand_factor` |
| `attention_every_k` | 4                      | *(Jamba only)* Insert attention every k layers       |
| `reduce_output`     | `"mean"`               | `"mean"` / `"sum"` / `"max"` / `"last"` / `None`     |

## When to prefer Mamba-2 over a Transformer

- Documents > 1 K tokens (Transformer attention is quadratic, Mamba-2 is linear)
- Compute or memory budget is tight (Mamba-2 ≈ 2-3× less memory than same-depth Transformer)
- Audio / timeseries features with very long windows

## When Jamba beats pure Mamba-2

- Tasks where cross-token position-insensitive dependencies matter (e.g., long-range co-reference)
- Medium-length sequences where the occasional attention layer helps without blowing the budget
