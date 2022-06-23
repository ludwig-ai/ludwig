import pytest
import torch

from ludwig.utils import output_feature_utils


def test_output_feature_utils():
    tensor_dict = {}
    output_feature_utils.set_output_feature_tensor(tensor_dict, "feature_1", "1", torch.Tensor([1]))
    output_feature_utils.set_output_feature_tensor(tensor_dict, "feature_1", "10", torch.Tensor([10]))
    output_feature_utils.set_output_feature_tensor(tensor_dict, "feature_2", "2", torch.Tensor([2]))
    output_feature_utils.set_output_feature_tensor(tensor_dict, "feature_2", "20", torch.Tensor([20]))

    assert list(tensor_dict.keys()) == ["feature_1::1", "feature_1::10", "feature_2::2", "feature_2::20"]
    assert output_feature_utils.get_output_feature_tensor(tensor_dict, "feature_1", "1") == torch.Tensor([1])
    assert list(output_feature_utils.get_single_output_feature_tensors(tensor_dict, "feature_1").keys()) == ["1", "10"]
    assert list(output_feature_utils.get_single_output_feature_tensors(tensor_dict, "feature_3").keys()) == []
    with pytest.raises(Exception):
        output_feature_utils.get_output_feature_tensor(tensor_dict, "feature_1", "2")
