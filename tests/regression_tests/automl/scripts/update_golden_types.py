#!/usr/bin/env python
"""This script updates all golden JSON files containing expected data types."""
import json

from ludwig.automl.automl import create_auto_config
from tests.regression_tests.automl.utils import get_dataset_golden_types_path, get_dataset_object, TEST_DATASET_REGISTRY


def write_json_files():
    for dataset_name in TEST_DATASET_REGISTRY:
        dataset_obj = get_dataset_object(dataset_name)
        dataset = dataset_obj.load(split=False)

        # NOTE: assuming type inference for input and output features is the same
        config = create_auto_config(
            dataset=dataset,
            target=[],
            time_limit_s=3600,
            tune_for_memory=False,
        )

        golden_types_path = get_dataset_golden_types_path(dataset_name)
        with open(golden_types_path, "w") as f:
            json.dump(config["input_features"], f, indent=4, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    write_json_files()
