"""Export Ludwig config JSON schema.

Usage:
    python -m ludwig.schema.export_schema [--model-type ecd|llm|combined] [--output FILE]
    ludwig export_schema [--model-type ecd|llm|combined] [--output FILE]

Generates a JSON Schema (Draft 7) for Ludwig config validation.
"""

import argparse
import json

from ludwig.config_validation.validation import get_schema
from ludwig.constants import MODEL_ECD, MODEL_LLM
from ludwig.globals import LUDWIG_VERSION

SCHEMA_BASE_URL = "https://ludwig-ai.github.io/schema"


def _strip_parameter_metadata(obj):
    """Recursively remove ``parameter_metadata`` keys from a schema dict.

    The Ludwig schema generator attaches ``parameter_metadata`` objects to
    every field (UI display hints, suggested values, etc.).  These are useful
    internally but add significant bloat to the published JSON Schema and are
    not relevant for validation or IDE auto-complete.
    """
    if isinstance(obj, dict):
        return {k: _strip_parameter_metadata(v) for k, v in obj.items() if k != "parameter_metadata"}
    if isinstance(obj, list):
        return [_strip_parameter_metadata(item) for item in obj]
    return obj


def export_schema(model_type: str = MODEL_ECD, *, strip_metadata: bool = True) -> dict:
    """Export the full Ludwig config JSON schema for a given model type."""
    schema = get_schema(model_type)
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["$id"] = f"{SCHEMA_BASE_URL}/ludwig-config-{model_type}.json"
    schema["title"] = f"Ludwig {model_type.upper()} Configuration"
    schema["description"] = f"Configuration schema for Ludwig {model_type.upper()} models (v{LUDWIG_VERSION})"
    if strip_metadata:
        schema = _strip_parameter_metadata(schema)
    return schema


def export_combined_schema(*, strip_metadata: bool = True) -> dict:
    """Export a combined schema that covers both ECD and LLM model types."""
    ecd_schema = get_schema(MODEL_ECD)
    llm_schema = get_schema(MODEL_LLM)

    # Merge properties from both schemas
    all_properties = {}
    all_properties.update(ecd_schema.get("properties", {}))
    all_properties.update(llm_schema.get("properties", {}))

    combined = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"{SCHEMA_BASE_URL}/ludwig-config.json",
        "title": "Ludwig Configuration",
        "description": f"Configuration schema for Ludwig models (v{LUDWIG_VERSION})",
        "type": "object",
        "properties": all_properties,
        "required": ["input_features", "output_features"],
        "additionalProperties": True,
    }
    if strip_metadata:
        combined = _strip_parameter_metadata(combined)
    return combined


def main(sys_argv=None):
    parser = argparse.ArgumentParser(description="Export Ludwig config JSON schema")
    parser.add_argument(
        "--model-type",
        choices=[MODEL_ECD, MODEL_LLM, "combined"],
        default="combined",
        help="Model type to export schema for (default: combined)",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (default: stdout)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include parameter_metadata in the output (default: stripped)",
    )
    args = parser.parse_args(sys_argv)

    strip_metadata = not args.full

    if args.model_type == "combined":
        schema = export_combined_schema(strip_metadata=strip_metadata)
    else:
        schema = export_schema(args.model_type, strip_metadata=strip_metadata)

    output = json.dumps(schema, indent=2, sort_keys=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
