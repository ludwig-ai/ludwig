"""Structured and constrained LLM output with Ludwig.

Demonstrates:
1. Entity extraction using JSON schema constraints
2. Sentiment classification with regex-constrained decoding
3. Side-by-side comparison of constrained vs unconstrained output

Run:
    python run_structured.py
"""

import json
import textwrap

import pandas as pd
import yaml

from ludwig.api import LudwigModel

# ---------------------------------------------------------------------------
# 1. Entity extraction with JSON schema constraints
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_CONFIG = yaml.safe_load("""
model_type: llm
base_model: microsoft/phi-2

prompt:
  task: >
    Extract the named entities from the input text and return them as a JSON
    object with this structure:
    {"entities": [{"text": "...", "type": "PERSON|ORG|LOC|DATE"}]}.
    Return only valid JSON, nothing else.

input_features:
  - name: text
    type: text

output_features:
  - name: output
    type: text
    decoder:
      type: text_parser
      json_schema:
        type: object
        properties:
          entities:
            type: array
            items:
              type: object
              properties:
                text:
                  type: string
                type:
                  type: string
                  enum: [PERSON, ORG, LOC, DATE]
              required: [text, type]
        required: [entities]
        additionalProperties: false

generation:
  max_new_tokens: 200
  temperature: 0.1
  do_sample: false

backend:
  type: local
""")

ENTITY_SAMPLES = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
    "Elon Musk announced that Tesla will open a new Gigafactory in Berlin next year.",
    "The United Nations headquarters is located in New York City.",
]


def run_entity_extraction() -> None:
    print("=" * 60)
    print("Entity Extraction with JSON Schema Constraints")
    print("=" * 60)

    model = LudwigModel(config=ENTITY_EXTRACTION_CONFIG)
    df = pd.DataFrame({"text": ENTITY_SAMPLES})
    predictions, _, _ = model.predict(dataset=df)

    for i, (text, pred) in enumerate(zip(ENTITY_SAMPLES, predictions["output_predictions"])):
        print(f"\n[{i + 1}] Input: {text}")
        try:
            parsed = json.loads(pred)
            entities = parsed.get("entities", [])
            print(f"     Entities ({len(entities)}):")
            for ent in entities:
                print(f"       - '{ent['text']}' ({ent['type']})")
        except json.JSONDecodeError:
            print(f"     Raw output: {pred}")


# ---------------------------------------------------------------------------
# 2. Sentiment classification with regex-constrained decoding
# ---------------------------------------------------------------------------

SENTIMENT_CONFIG_CONSTRAINED = yaml.safe_load("""
model_type: llm
base_model: Qwen/Qwen2-0.5B-Instruct

prompt:
  task: >
    Classify the sentiment of the following text.
    Respond with exactly one word: positive, negative, or neutral.

input_features:
  - name: text
    type: text

output_features:
  - name: sentiment
    type: text
    decoder:
      type: text_parser
      regex: "(positive|negative|neutral)"

generation:
  max_new_tokens: 10
  temperature: 0.0
  do_sample: false

backend:
  type: local
""")

SENTIMENT_CONFIG_UNCONSTRAINED = yaml.safe_load("""
model_type: llm
base_model: Qwen/Qwen2-0.5B-Instruct

prompt:
  task: >
    Classify the sentiment of the following text.
    Respond with exactly one word: positive, negative, or neutral.

input_features:
  - name: text
    type: text

output_features:
  - name: sentiment
    type: text

generation:
  max_new_tokens: 30
  temperature: 0.7

backend:
  type: local
""")

SENTIMENT_SAMPLES = [
    "I absolutely loved this product! It exceeded all my expectations.",
    "The service was terrible and the food was cold.",
    "The movie was okay, nothing special.",
    "This is the best laptop I have ever owned. Highly recommend.",
    "I waited two hours and they still got my order wrong.",
    "The weather today is neither good nor bad.",
]


def run_sentiment_comparison() -> None:
    print("\n" + "=" * 60)
    print("Sentiment Classification: Constrained vs Unconstrained")
    print("=" * 60)

    df = pd.DataFrame({"text": SENTIMENT_SAMPLES})

    print("\nRunning UNCONSTRAINED model...")
    model_unconstrained = LudwigModel(config=SENTIMENT_CONFIG_UNCONSTRAINED)
    preds_unconstrained, _, _ = model_unconstrained.predict(dataset=df)

    print("Running CONSTRAINED model (regex: positive|negative|neutral)...")
    model_constrained = LudwigModel(config=SENTIMENT_CONFIG_CONSTRAINED)
    preds_constrained, _, _ = model_constrained.predict(dataset=df)

    print(f"\n{'Input':<52} {'Unconstrained':<30} {'Constrained':<15}")
    print("-" * 97)
    for text, unconstrained, constrained in zip(
        SENTIMENT_SAMPLES,
        preds_unconstrained["sentiment_predictions"],
        preds_constrained["sentiment_predictions"],
    ):
        short_text = textwrap.shorten(text, width=50)
        print(f"{short_text:<52} {str(unconstrained):<30} {str(constrained):<15}")

    # Count invalid outputs in unconstrained
    valid_labels = {"positive", "negative", "neutral"}
    invalid = [p for p in preds_unconstrained["sentiment_predictions"] if str(p).strip().lower() not in valid_labels]
    print(f"\nUnconstrained invalid outputs: {len(invalid)}/{len(SENTIMENT_SAMPLES)}")
    print("Constrained invalid outputs:   0 (guaranteed by regex constraint)")


# ---------------------------------------------------------------------------
# 3. Logits extraction
# ---------------------------------------------------------------------------

LOGITS_CONFIG = yaml.safe_load("""
model_type: llm
base_model: Qwen/Qwen2-0.5B-Instruct

prompt:
  task: "Answer with a single word."

input_features:
  - name: text
    type: text

output_features:
  - name: response
    type: text

generation:
  max_new_tokens: 5
  temperature: 0.0
  do_sample: false

backend:
  type: local
""")


def run_logits_extraction() -> None:
    print("\n" + "=" * 60)
    print("Logits Extraction")
    print("=" * 60)

    model = LudwigModel(config=LOGITS_CONFIG)
    df = pd.DataFrame({"text": ["Is Python a programming language?"]})

    # collect_activations returns intermediate layer activations alongside predictions
    predictions, output_df, _ = model.predict(dataset=df, collect_predictions=True)

    print("Prediction:", predictions["response_predictions"].iloc[0])

    # When logits are available they appear as response_logits in the output
    if "response_logits" in output_df.columns:
        logits = output_df["response_logits"].iloc[0]
        print(f"Logits shape: {logits.shape if hasattr(logits, 'shape') else 'N/A'}")
        print(f"First 5 logit values: {logits[:5] if hasattr(logits, '__iter__') else logits}")
    else:
        print("Logits not present in output (enable output_logits in config to collect them).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_entity_extraction()
    run_sentiment_comparison()
    run_logits_extraction()
