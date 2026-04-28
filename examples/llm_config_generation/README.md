# LLM-Driven Config Generation

> **Note:** This feature requires PR #4092 to be merged into Ludwig, or `pip install ludwig>=0.14`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/llm_config_generation/llm_config_generation.ipynb)

## What is this?

Ludwig's config generation feature lets you describe your machine learning task in plain English and receive a fully validated Ludwig configuration file in return. An LLM (Claude or GPT-4) interprets your description, maps column names to Ludwig feature types, selects an appropriate model architecture, and emits a config dict that passes Ludwig's Pydantic schema validation before it ever reaches your code.

This is particularly useful for:

- **New users** who are unfamiliar with Ludwig's YAML schema and want a working starting point.
- **Rapid prototyping** — describe the task, inspect the generated config, tweak if needed, and run.
- **Multi-task problems** — describing simultaneous outputs (e.g. classify + regress) is often easier in prose than in YAML.

## Prerequisites

You need an API key for at least one of the supported backends:

| Backend            | Environment variable |
| ------------------ | -------------------- |
| Anthropic (Claude) | `ANTHROPIC_API_KEY`  |
| OpenAI (GPT)       | `OPENAI_API_KEY`     |

The library reads the key automatically from the environment. You can also pass `api_key=` explicitly.

Install the required packages:

```bash
pip install "ludwig>=0.14" anthropic   # for Claude
# or
pip install "ludwig>=0.14" openai      # for GPT
```

## Quick start

```python
import os
import yaml
from ludwig.config_generation import generate_config  # requires PR #4092 / ludwig>=0.14

config = generate_config(
    "I have customer data with age, income, and purchase history. "
    "I want to predict churn (binary) and lifetime value (number).",
    model="claude-sonnet-4-20250514",
    # api_key is read from ANTHROPIC_API_KEY by default
    validate=True,
)

print(yaml.dump(config, default_flow_style=False))
```

You can also use an OpenAI model by passing its name:

```python
config = generate_config(
    "Predict apartment rent price from sqft, bedrooms, and neighborhood.",
    model="gpt-4o",
    validate=True,
)
```

The backend is chosen automatically based on whether the model name starts with `"claude"` or `"gpt"`.

## Files

| File                          | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| `README.md`                   | This file                                               |
| `llm_config_generation.ipynb` | Interactive walkthrough notebook                        |
| `generate_and_train.py`       | Standalone CLI script — describe a task, confirm, train |

## Running the standalone script

```bash
# Use the default task description
python generate_and_train.py

# Or pass your own description
python generate_and_train.py "predict house price from bedrooms, sqft, and location"

# Use a specific model
python generate_and_train.py --model gpt-4o "classify email sentiment as positive, neutral, or negative"
```

## Tips for writing good task descriptions

- **Name your columns** — "age, income, and purchase_count" is more actionable than "some user features".
- **State the target and its type** — "predict churn (binary)" or "predict revenue (continuous number)".
- **Mention modalities** — "text product description and tabular price, category" helps Ludwig pick the right encoder.
- **Include rough dataset size** — "~50 k rows" lets the LLM suggest appropriate model complexity.
- **Describe multi-output tasks explicitly** — "simultaneously predict price (regression) and category (classification)".
