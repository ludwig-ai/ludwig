import torch
from typing import Dict


def get_feature_concat_name(feature_name: str, tensor_name: str):
    return feature_name + '::' + tensor_name


def get_tensor_name_from_concat_name(concat_name: str):
    return concat_name.split('::')[-1]


def get_feature_name_from_concat_name(concat_name: str):
    return concat_name.split('::')[:-1]


def get_single_output_feature_tensors(
        output_feature_dict: Dict[str, torch.Tensor],
        feature_name: str) -> Dict[str, torch.Tensor]:
    single_output_feature_tensors = {}
    for concat_name, tensor in output_feature_dict.items():
        if get_feature_name_from_concat_name(concat_name) == feature_name:
            single_output_feature_tensors[get_tensor_name_from_concat_name(
                concat_name)] = tensor
    return single_output_feature_tensors


def get_output_feature_tensor(
        output_dict: Dict[str, torch.Tensor],
        feature_name: str,
        tensor_name: str) -> torch.Tensor:
    return output_dict[get_feature_concat_name(feature_name, tensor_name)]


def set_output_feature_tensor(
        output_dict: Dict[str, torch.Tensor],
        feature_name: str,
        tensor_name: str, tensor: torch.Tensor):
    output_dict[get_feature_concat_name(feature_name, tensor_name)] = tensor
