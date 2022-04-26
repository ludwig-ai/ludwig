#!/usr/bin/env python
"""This script updates all golden JSON files containing expected data types."""
import json

from ludwig.automl.automl import create_auto_config
from tests.regression_tests.automl.type_inference_test_utils import get_dataset_golden_types_path, REGISTRY

if __name__ == "__main__":
    for dataset_module in REGISTRY:
        dataset = dataset_module.load(split=False)

        # NOTE: assuming type inference for input and output features is the same
        config = create_auto_config(
            dataset=dataset,
            target=[],
            time_limit_s=3600,
            tune_for_memory=False,
        )

        golden_types_path = get_dataset_golden_types_path(dataset_module)
        with open(golden_types_path, "w") as f:
            json.dump(config["input_features"], f, indent=4, sort_keys=True)
