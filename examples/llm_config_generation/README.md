# LLM-Driven Ludwig Config Generation

> **Requires Ludwig 0.15 / PR #4092 (future-capabilities branch).**

`ludwig generate_config` turns a natural-language description of an ML task into a valid
Ludwig YAML config. It does this in three steps:

1. Extract a compact representation of Ludwig's Pydantic schema (feature types, combiner
   registry, trainer fields, loss balancing options, presets).
1. Feed your task description plus that schema context to an LLM.
1. Parse the LLM's YAML response and validate it against Ludwig's real Pydantic schema.
   Invalid configs are rejected before they ever reach the training loop.

## Requirements

- An Anthropic or OpenAI API key exported as an env var
  (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).
- Ludwig 0.15 with the generate_config module.

The default model is `claude-sonnet-4-20250514`. Pass `--model gpt-4o` (or similar) to use
an OpenAI model instead.

## Usage

### CLI

```bash
export ANTHROPIC_API_KEY="..."
ludwig generate_config "I have a CSV with columns age (number), income (number), \
education (text), and marital_status (category). I want to predict whether someone \
will default on a loan (binary)." \
  --output loan_default.yaml
```

Use `-` or pipe stdin if your description is long:

```bash
cat task.txt | ludwig generate_config --output config.yaml
```

Flags:

| Flag            | Default                    | Description                        |
| --------------- | -------------------------- | ---------------------------------- |
| `description`   | (required, or stdin)       | Natural-language description       |
| `--model`       | `claude-sonnet-4-20250514` | LLM model to use                   |
| `--api_key`     | env var                    | API key override                   |
| `--output / -o` | stdout                     | Output file path                   |
| `--no-validate` | off                        | Skip Pydantic validation of result |

### Python API

```python
from ludwig.config_generation import generate_config

config = generate_config(
    task_description=(
        "I have customer data with age, income, and purchase_history columns. "
        "I want to predict churn (binary) and lifetime_value (number) jointly."
    ),
    model="claude-sonnet-4-20250514",  # or "gpt-4o"
    api_key=None,  # reads from env
    validate=True,  # reject invalid configs
)

import yaml

print(yaml.dump(config))
```

## End-to-end example: generate and train

`generate_and_train.py` takes a task description, calls `generate_config`, writes the
config to disk, and kicks off training:

```bash
pip install ludwig anthropic  # or openai if using GPT
export ANTHROPIC_API_KEY="..."
python generate_and_train.py
```

The script uses the [UCI Adult (census income) dataset](https://archive.ics.uci.edu/dataset/2/adult)
already shipped with Ludwig via `ludwig.datasets.adult_census_income`.

## Writing good task descriptions

The LLM's output is only as good as your prompt. The descriptions that work best have:

- **Each column listed by name and type** — include (number), (category), (text), (binary),
  (image), (sequence), (datetime) hints inline.
- **A clear prediction target** — "I want to predict <column> (<type>)".
- **Optional constraints** — "fast training", "use a pretrained text encoder",
  "apply 4-bit quantization", "use DPO for alignment".

Example of a strong description:

```
I have a parquet file with:
  - age (number)
  - occupation (category, 14 unique values)
  - review_text (text, ~500 tokens typical)
  - profile_image (image, 224x224 RGB)
  - num_purchases (number)

I want to predict willingness_to_pay (number, regression) and
recommended_tier (category, 3 classes) jointly. The dataset has 50k rows,
training should be fast — prefer the medium_quality preset and stacked_cnn
image encoder with trainable: false.
```

## When the LLM gets it wrong

The strict Pydantic validation catches most errors (wrong feature types, misspelled combiner
names, invalid trainer fields). When `validate=True` (the default) the generation raises
`ValueError` and surfaces the validation error. Common fixes:

- Rewrite your description to be more explicit about column names and types.
- Try the same prompt with a different model (`--model gpt-4o`).
- Use `--no-validate` to inspect the raw LLM output for debugging, then hand-edit.

## Files

| File                      | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `generate_and_train.py`   | Generate a config from prompt and train on adult income |
| `example_description.txt` | Example natural-language task description               |
| `README.md`               | This file                                               |

## Notes

This feature does **not** replace careful model design. Use it to scaffold a first config
quickly, then iterate — adjust preprocessing, tune the trainer, swap encoders based on what
your validation metrics actually show.
