from copy import deepcopy
from typing import Dict

import numpy as np
import pytest
import torch

from ludwig.features.number_feature import _OutlierReplacer, NumberInputFeature
from ludwig.schema.features.number_feature import ECDNumberInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def number_config():
    return {"name": "number_column_name", "type": "number"}


def test_number_input_feature(
    number_config: Dict,
) -> None:
    # setup image input feature definition
    number_def = deepcopy(number_config)

    # pickup any other missing parameters
    defaults = ECDNumberInputFeatureConfig().to_dict()
    set_def = merge_dict(defaults, number_def)

    # ensure no exceptions raised during build
    number_config, _ = load_config_with_kwargs(ECDNumberInputFeatureConfig, set_def)
    input_feature_obj = NumberInputFeature(number_config).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = input_feature_obj.create_sample_input(batch_size=BATCH_SIZE)
    assert input_tensor.shape == torch.Size([BATCH_SIZE])
    assert input_tensor.dtype == torch.float32

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)


def test_outlier_replacer():
    replacer = _OutlierReplacer(
        {"mean": 50, "std": 30, "preprocessing": {"outlier_threshold": 2.0, "computed_outlier_fill_value": 42}}
    )

    t = torch.from_numpy(np.array([10, 20, 1000, -500, 80], dtype=np.float32))
    t_out_expected = torch.from_numpy(np.array([10, 20, 42, 42, 80], dtype=np.float32))

    t_out = replacer(t)
    assert torch.equal(t_out, t_out_expected)
