import json

import pytest

from tests.regression_tests.automl.utils import get_dataset_golden_types_path, get_dataset_object, TEST_DATASET_REGISTRY

try:
    from ludwig.automl import create_auto_config
except ImportError:
    pass


@pytest.mark.slow
@pytest.mark.distributed  # ludwig.automl has a dependency on ray
@pytest.mark.parametrize("dataset_name", TEST_DATASET_REGISTRY)
def test_auto_type_inference_regression(dataset_name):
    golden_types_path = get_dataset_golden_types_path(dataset_name)
    with open(golden_types_path) as f:
        golden_types = json.load(f)

    dataset_obj = get_dataset_object(dataset_name)
    dataset = dataset_obj.load(split=False)

    # NOTE: assuming type inference for input and output features is the same
    config = create_auto_config(
        dataset=dataset,
        target=[],
        time_limit_s=3600,
    )

    assert golden_types == config["input_features"]
