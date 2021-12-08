"""Utilities used for managing output feature dicts."""

from typing import Dict

import torch


def get_feature_concat_name(feature_name: str, tensor_name: str) -> str:
    return feature_name + "::" + tensor_name


def get_tensor_name_from_concat_name(concat_name: str) -> str:
    return concat_name.split("::")[-1]


def get_feature_name_from_concat_name(concat_name: str) -> str:
    return "::".join(concat_name.split("::")[:-1])


def get_single_output_feature_tensors(
    output_feature_dict: Dict[str, torch.Tensor], feature_name: str
) -> Dict[str, torch.Tensor]:
    """Returns a map of tensors related to the given feature_name."""
    single_output_feature_tensors = {}
    for concat_name, tensor in output_feature_dict.items():
        if get_feature_name_from_concat_name(concat_name) == feature_name:
            single_output_feature_tensors[get_tensor_name_from_concat_name(concat_name)] = tensor
    return single_output_feature_tensors


def get_output_feature_tensor(
    output_dict: Dict[str, torch.Tensor], feature_name: str, tensor_name: str
) -> torch.Tensor:
    """Returns a tensor related for the given feature_name and tensor_name."""
    concat_name = get_feature_concat_name(feature_name, tensor_name)
    if concat_name not in output_dict:
        raise ValueError(
            f"Could not find {tensor_name} for {feature_name} in the output_dict with keys: {output_dict.keys()}"
        )
    return output_dict[get_feature_concat_name(feature_name, tensor_name)]


def set_output_feature_tensor(
    output_dict: Dict[str, torch.Tensor], feature_name: str, tensor_name: str, tensor: torch.Tensor
):
    """Adds tensor for the given feature_name and tensor_name to the tensor dict."""
    output_dict[get_feature_concat_name(feature_name, tensor_name)] = tensor
