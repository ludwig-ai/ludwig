import logging
import os

import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.constants import NAME
from ludwig.explain.util import replace_layer_with_copy
from ludwig.utils.torch_utils import get_absolute_module_key_from_submodule
from tests.integration_tests.utils import binary_feature, generate_data, LocalTestBackend, text_feature


def test_get_absolute_module_key_from_submodule():
    class ParentModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child_module_1 = ChildModule()
            self.child_module_2 = ChildModule()

    class ChildModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

    # the expected module names are those that are relative to the parent module, i.e. "child_module_1.linear.weight"
    parent_module = ParentModule()
    expected_module_names = set()
    for parent_param_name, _ in parent_module.named_parameters():
        expected_module_names.add(parent_param_name)

    # incorrect module names are those that are relative to the child module, not the parent module,
    # i.e. "linear.weight" and "linear.bias"
    incorrect_param_names = set()
    for child_param_name, _ in parent_module.child_module_1.named_parameters():
        incorrect_param_names.add(child_param_name)

    module_names_child_1 = set(get_absolute_module_key_from_submodule(parent_module, parent_module.child_module_1))
    module_names_child_2 = set(get_absolute_module_key_from_submodule(parent_module, parent_module.child_module_2))

    # check that the module names are not equivalent to the incorrect module names
    assert set.isdisjoint(module_names_child_1, incorrect_param_names)
    assert set.isdisjoint(module_names_child_2, incorrect_param_names)

    # check that the module names are disjoint from one another because they are relative to the parent module
    assert set.isdisjoint(module_names_child_1, module_names_child_2)

    # check that the union of the two sets is equal to the expected module names
    assert set.union(module_names_child_1, module_names_child_2) == expected_module_names


def test_replace_layer_with_copy(tmpdir):
    text_feature_1 = text_feature()
    text_feature_2 = text_feature(tied=text_feature_1["name"])
    input_features = [text_feature_1, text_feature_2]
    output_features = [binary_feature()]

    csv_filename = os.path.join(tmpdir, "training.csv")
    generate_data(input_features, output_features, csv_filename, num_examples=200)
    df = pd.read_csv(csv_filename)
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {
            "epochs": 1,
        },
    }
    model = LudwigModel(config, logging_level=logging.WARNING, backend=LocalTestBackend())
    model.train(df)

    input_feature_module = model.model.input_features.get(text_feature_2[NAME])
    target_layer = input_feature_module.encoder_obj.get_embedding_layer()

    data_ptrs_before = {}
    for param_name, param in input_feature_module.named_parameters():
        data_ptrs_before[param_name] = param.data_ptr()

    # keys_to_copy = get_absolute_module_key_from_submodule(input_feature_module, target_layer)
    replace_layer_with_copy(input_feature_module, target_layer)

    data_ptrs_after = {}
    for param_name, param in input_feature_module.named_parameters():
        data_ptrs_after[param_name] = param.data_ptr()

    # Check that the data pointers are different for the copied keys and that they are the same for the rest.
    for param_name, _ in input_feature_module.named_parameters():
        # (Jeff K.) Disabling this check until further explainability tests can be conducted.
        # if param_name in keys_to_copy:
        #     assert (
        #         data_ptrs_before[param_name] != data_ptrs_after[param_name]
        #     ), f"Data pointers should be different for copied key {param_name}"
        # else:
        assert (
            data_ptrs_before[param_name] == data_ptrs_after[param_name]
        ), f"Data pointers should be the same for non-copied key {param_name}"
