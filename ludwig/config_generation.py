"""LLM-driven config generation: describe your ML task in English, get a Ludwig config.

Uses the model's JSON Schema as context for an LLM to generate valid Ludwig configs.
The generated config is validated with strict Pydantic validation before returning.

Usage:
    from ludwig.config_generation import generate_config

    config = generate_config(
        "I have a CSV with columns: age (number), income (number), education (text), "
        "and I want to predict whether someone will default on a loan (binary)."
    )
    # Returns a valid Ludwig config dict
"""

import json
import logging

logger = logging.getLogger(__name__)


def get_ludwig_schema_context() -> str:
    """Get a compact JSON Schema description of Ludwig's config for LLM context."""
    try:
        from ludwig.schema.model_types.ecd import ECDModelConfig

        ECDModelConfig.model_json_schema()  # Validate schema is available
        # Extract just the key parts to fit in context window
        return json.dumps(
            {
                "description": "Ludwig declarative ML config. "
                "Specify input_features, output_features, combiner, trainer.",
                "input_feature_types": [
                    "number",
                    "category",
                    "binary",
                    "text",
                    "image",
                    "audio",
                    "sequence",
                    "set",
                    "vector",
                    "timeseries",
                    "date",
                    "h3",
                    "bag",
                ],
                "output_feature_types": ["number", "category", "binary", "text", "sequence", "set", "vector"],
                "combiner_types": [
                    "concat",
                    "transformer",
                    "ft_transformer",
                    "cross_attention",
                    "perceiver",
                    "gated_fusion",
                    "tabnet",
                    "tabtransformer",
                    "comparator",
                    "project_aggregate",
                    "sequence",
                    "sequence_concat",
                ],
                "encoder_types_number": ["passthrough", "dense", "ple", "periodic"],
                "encoder_types_category": ["dense", "sparse", "onehot", "passthrough"],
                "encoder_types_text": [
                    "auto_transformer",
                    "bert",
                    "gpt2",
                    "parallel_cnn",
                    "stacked_cnn",
                    "stacked_parallel_cnn",
                    "rnn",
                    "cnnrnn",
                    "transformer",
                ],
                "loss_balancing": ["none", "log_transform", "uncertainty", "famo", "gradnorm"],
                "trainer_type_ecd": "trainer (epochs, batch_size, "
                "learning_rate, optimizer, early_stop, loss_balancing)",
                "trainer_type_llm": "finetune, dpo, kto, orpo, grpo, none",
                "presets": ["medium_quality", "high_quality", "best_quality"],
                "example_config": {
                    "input_features": [
                        {"name": "text_col", "type": "text", "encoder": {"type": "auto_transformer"}},
                        {"name": "num_col", "type": "number"},
                    ],
                    "output_features": [{"name": "target", "type": "category"}],
                    "combiner": {"type": "concat"},
                    "trainer": {"epochs": 50, "batch_size": 128},
                },
            },
            indent=2,
        )
    except Exception:
        return "{}"


def generate_config(
    task_description: str,
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
    validate: bool = True,
) -> dict:
    """Generate a Ludwig config from a natural language task description.

    Uses an LLM to translate the description into a valid Ludwig YAML config.
    The generated config is validated against Ludwig's Pydantic schema.

    Args:
        task_description: Natural language description of the ML task.
            Example: "I have customer data with age, income, and purchase history.
            I want to predict churn (binary) and lifetime value (number)."
        model: LLM model to use for generation.
        api_key: API key for the LLM provider. If None, reads from environment.
        validate: If True, validate the generated config against Ludwig's schema.

    Returns:
        Dict with a valid Ludwig config.

    Raises:
        ValueError if the generated config is invalid and validate=True.
    """
    schema_context = get_ludwig_schema_context()

    prompt = f"""You are a Ludwig ML framework expert. Generate a valid Ludwig YAML config for this task.

Ludwig Config Schema Context:
{schema_context}

Task Description:
{task_description}

Generate ONLY a valid JSON config (no markdown, no explanation). The config must have:
- input_features: list of dicts with name, type, and optional encoder
- output_features: list of dicts with name and type
- combiner: dict with type and parameters
- trainer: dict with epochs, batch_size, learning_rate

Choose appropriate feature types, encoders, and combiner based on the task."""

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        config_str = response.content[0].text
    except ImportError:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model if "gpt" in model else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            config_str = response.choices[0].message.content
        except ImportError:
            raise ImportError(
                "Either 'anthropic' or 'openai' package is required for config generation. "
                "Install with: pip install anthropic  OR  pip install openai"
            )

    # Parse JSON from response (handle markdown code blocks)
    config_str = config_str.strip()
    if config_str.startswith("```"):
        lines = config_str.split("\n")
        config_str = "\n".join(lines[1:-1])

    config = json.loads(config_str)

    if validate:
        try:
            from ludwig.schema.model_types.base import ModelConfig

            ModelConfig.from_dict(config)
            logger.info("Generated config validated successfully")
        except Exception as e:
            raise ValueError(f"Generated config failed validation: {e}") from e

    return config
