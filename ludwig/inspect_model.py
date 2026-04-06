"""CLI for model inspection -- ``ludwig inspect``."""

import argparse
import json
import logging
import sys

from ludwig.api import LudwigModel

logger = logging.getLogger(__name__)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="Inspect a trained Ludwig model",
        prog="ludwig inspect",
    )
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained model directory")
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Show detailed weight tensor info",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Show approximate feature importance from encoder weights",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args(sys_argv)

    model = LudwigModel.load(args.model_path)

    from ludwig.model_inspector import ModelInspector

    inspector = ModelInspector(
        model=model.model,
        config=model.config,
        training_set_metadata=model.training_set_metadata,
    )

    summary = inspector.model_summary()

    if args.output_json:
        output = {"summary": summary}
        if args.weights:
            output["weights"] = inspector.collect_weights()
        if args.importance:
            output["feature_importance"] = inspector.feature_importance_proxy()
        print(json.dumps(output, indent=2))
    else:
        print(f"\nModel Summary")
        print(f"{'=' * 50}")
        print(f"  Model type:           {summary['model_type']}")
        print(f"  Combiner:             {summary['combiner_type']}")
        print(f"  Input features:       {summary['num_input_features']}")
        print(f"  Output features:      {summary['num_output_features']}")
        print(f"  Total parameters:     {summary['total_parameters']:,}")
        print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"  Frozen parameters:    {summary['frozen_parameters']:,}")
        print(f"  Model size:           {summary['model_size_mb']:.2f} MB")
        print()

        if args.weights:
            weights = inspector.collect_weights()
            print(f"Weights ({len(weights)} tensors)")
            print(f"{'=' * 50}")
            for w in weights:
                grad = "trainable" if w["requires_grad"] else "frozen"
                print(f"  {w['name']}: {w['shape']} ({w['num_elements']:,} params, {grad})")
            print()

        if args.importance:
            importance = inspector.feature_importance_proxy()
            if importance:
                print(f"Feature Importance (approximate)")
                print(f"{'=' * 50}")
                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for name, score in sorted_imp:
                    bar = "#" * int(score * 30)
                    print(f"  {name:30s} {score:.4f} {bar}")
            else:
                print("  No input features found for importance estimation")
            print()
