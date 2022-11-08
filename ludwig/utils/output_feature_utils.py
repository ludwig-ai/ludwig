"""Utilities used for managing output feature dicts."""

from typing import Dict, List

import numpy as np
import torch

from ludwig.utils.torch_utils import sequence_length_3D, sequence_mask


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


def concat_dependencies(
    feature_name: str,
    dependencies: List[str],
    dependency_reducers: torch.ModuleDict,
    combiner_hidden_state: torch.Tensor,
    other_output_feature_states: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Concatenates combiner_hidden_state with other output feature hidden states based on listed dependencies."""
    # No dependencies.
    if not dependencies:
        return combiner_hidden_state

    dependency_hidden_states = []
    for feature_name in dependencies:
        # The dependent feature should be present since ECD does a topological sort over output features.
        feature_hidden_state = other_output_feature_states[feature_name]

        # This feature is sequential.
        if len(combiner_hidden_state.shape) > 2:
            if len(feature_hidden_state.shape) > 2:
                # The dependent feature is also sequential.
                # matrix matrix -> concat
                assert combiner_hidden_state.shape[1] == feature_hidden_state.shape[1]
                dependency_hidden_states.append(feature_hidden_state)
            else:
                # The dependent feature is not sequential.
                # matrix vector -> tile concat
                sequence_max_length = combiner_hidden_state.shape[1]
                multipliers = (1, sequence_max_length, 1)
                tiled_representation = torch.tile(torch.unsqueeze(feature_hidden_state, 1), multipliers)

                sequence_length = sequence_length_3D(combiner_hidden_state)
                mask = sequence_mask(sequence_length, sequence_max_length)
                tiled_representation = torch.mul(
                    tiled_representation,
                    mask[:, :, np.newaxis].type(torch.float32),
                )

                dependency_hidden_states.append(tiled_representation)

        else:
            # This feature is not sequential.
            if len(feature_hidden_state.shape) > 2:
                # The dependent feature is sequential.
                # vector matrix -> reduce concat
                reducer = dependency_reducers[feature_name]
                dependency_hidden_states.append(reducer(feature_hidden_state))
            else:
                # The dependent feature is not sequential.
                # vector vector -> concat
                dependency_hidden_states.append(feature_hidden_state)

    try:
        hidden = torch.cat([combiner_hidden_state] + dependency_hidden_states, dim=-1)
    except Exception as e:
        raise ValueError(
            f"Shape mismatch {e} while concatenating dependent features of {feature_name}: "
            f"{dependencies}. Concatenating the feature activations tensor {combiner_hidden_state} "
            f"with activation tensors of dependencies: {dependency_hidden_states}. The error is "
            "likely due to a mismatch of the second dimension (sequence length) or a "
            "difference in ranks. Likely solutions are setting the maximum_sequence_length "
            "of all sequential features to be the same,  or reduce the output of some "
            "features, or disabling the bucketing setting bucketing_field to None / null, "
            "as activating it will reduce the length of the field the bucketing is "
            "performed on."
        )
    return hidden
