"""Export Ludwig config JSON schema.

Usage:
    python -m ludwig.schema.export_schema [--model-type ecd|llm] [--output FILE]

Generates a JSON Schema (Draft 7) for Ludwig config validation.
"""

import argparse
import json

from ludwig.config_validation.validation import get_schema
from ludwig.constants import MODEL_ECD, MODEL_LLM
from ludwig.globals import LUDWIG_VERSION


def export_schema(model_type: str = MODEL_ECD) -> dict:
    """Export the full Ludwig config JSON schema for a given model type."""
    schema = get_schema(model_type)
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["$id"] = f"https://ludwig-ai.github.io/schema/ludwig-config-{model_type}.json"
    schema["title"] = f"Ludwig {model_type.upper()} Configuration"
    schema["description"] = f"Configuration schema for Ludwig {model_type.upper()} models (v{LUDWIG_VERSION})"
    return schema


def export_combined_schema() -> dict:
    """Export a combined schema that covers both ECD and LLM model types."""
    ecd_schema = get_schema(MODEL_ECD)
    llm_schema = get_schema(MODEL_LLM)

    # Merge properties from both schemas
    all_properties = {}
    all_properties.update(ecd_schema.get("properties", {}))
    all_properties.update(llm_schema.get("properties", {}))

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://ludwig-ai.github.io/schema/ludwig-config.json",
        "title": "Ludwig Configuration",
        "description": f"Configuration schema for Ludwig models (v{LUDWIG_VERSION})",
        "type": "object",
        "properties": all_properties,
        "required": ["input_features", "output_features"],
        "additionalProperties": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Ludwig config JSON schema")
    parser.add_argument(
        "--model-type",
        choices=[MODEL_ECD, MODEL_LLM, "combined"],
        default="combined",
        help="Model type to export schema for (default: combined)",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.model_type == "combined":
        schema = export_combined_schema()
    else:
        schema = export_schema(args.model_type)

    output = json.dumps(schema, indent=2, sort_keys=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
