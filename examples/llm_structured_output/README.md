# Structured and Constrained LLM Output

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/llm_structured_output/structured_output.ipynb)

## Why constrained decoding matters

Large language models are trained to produce fluent text, but they have no built-in guarantee that their output follows a particular format. When you ask an LLM to return JSON, classify into one of three labels, or follow a grammar, it may occasionally hallucinate, add extra text, or produce a structurally invalid response.

**Constrained decoding** solves this by modifying the token sampling process at inference time. A constraint (JSON schema, regular expression, or context-free grammar) is compiled into a set of masks that are applied to the model's logit distribution at each step. Tokens that would violate the constraint are assigned negative infinity logit, so the model can only ever produce valid output.

Ludwig supports three forms of constrained output:

| Constraint type | Use case                                        | Config key            |
| --------------- | ----------------------------------------------- | --------------------- |
| JSON schema     | Structured data extraction, tool-call responses | `decoder.json_schema` |
| Regex           | Classification, fixed-format fields             | `decoder.regex`       |
| Grammar (EBNF)  | Complex structured formats                      | `decoder.grammar`     |

## Quick start

```bash
pip install "ludwig[llm]"
```

### Entity extraction (JSON schema)

```bash
python run_structured.py
```

Or use the Ludwig API directly with one of the provided configs:

```python
import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel(config="config_json_schema.yaml")
preds, _, _ = model.predict(dataset=pd.DataFrame({"text": ["Apple was founded by Steve Jobs in Cupertino."]}))
print(preds["output_predictions"].iloc[0])
# -> {"entities": [{"text": "Apple", "type": "ORG"}, ...]}
```

### Sentiment classification (regex)

```python
model = LudwigModel(config="config_constrained.yaml")
preds, _, _ = model.predict(dataset=pd.DataFrame({"text": ["I loved this product!"]}))
print(preds["sentiment_predictions"].iloc[0])
# -> positive
```

## Files

| File                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `structured_output.ipynb` | Interactive notebook (Colab-compatible)                      |
| `config_json_schema.yaml` | Ludwig config for JSON schema entity extraction              |
| `config_constrained.yaml` | Ludwig config for regex-constrained sentiment classification |
| `run_structured.py`       | Standalone script showing all three features                 |

## Models used

Both configs use freely available models that fit on a free Colab GPU (T4, 16 GB):

- [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) — 2.7 B parameters, JSON schema examples
- [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) — 0.5 B parameters, classification examples

## Further reading

- [Structured and Constrained LLM Output — Ludwig User Guide](../../docs/user_guide/llms/structured_output.md)
- [Ludwig LLM configuration reference](../../docs/configuration/large_language_model.md)
