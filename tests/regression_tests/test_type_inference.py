import json

import pytest

from ludwig.automl.automl import create_auto_config
from tests.regression_tests.type_inference_test_utils import get_dataset_golden_types_path, REGISTRY


@pytest.mark.parametrize("dataset_module", REGISTRY)
def test_ludwig_auto_config(dataset_module):
    golden_types_path = get_dataset_golden_types_path(dataset_module)
    with open(golden_types_path) as f:
        golden_types = json.load(f)

    dataset = dataset_module.load(split=False)

    # NOTE: assuming type inference for input and output features is the same
    config = create_auto_config(
        dataset=dataset,
        target=[],
        time_limit_s=3600,
        tune_for_memory=False,
    )

    assert golden_types == config["input_features"]
