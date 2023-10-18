import logging

import pytest
import torch

from ludwig.combiners.combiners import ConcatCombiner
from ludwig.constants import CATEGORY, DECODER, NUMBER, SEQUENCE, TYPE
from ludwig.models.base import BaseModel
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.model_config import ModelConfig
from ludwig.utils import output_feature_utils
from tests.integration_tests.utils import generate_output_features_with_dependencies, number_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

BATCH_SIZE = 16
SEQ_SIZE = 12
HIDDEN_SIZE = 128
OTHER_HIDDEN_SIZE = 32
OTHER_HIDDEN_SIZE2 = 64


# unit test for dependency concatenation
# tests both single and multiple dependencies
@pytest.mark.parametrize(
    "dependent_hidden_shape2",
    [
        None,
        [BATCH_SIZE, OTHER_HIDDEN_SIZE2],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE2],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE],
    ],
)
@pytest.mark.parametrize(
    "dependent_hidden_shape", [[BATCH_SIZE, OTHER_HIDDEN_SIZE], [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]]
)
@pytest.mark.parametrize("hidden_shape", [[BATCH_SIZE, HIDDEN_SIZE], [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE]])
@pytest.mark.parametrize(
    # todo: re-add 'attention' after further research in implication of torch
    #       migration
    "reduce_dependencies",
    ["sum", "mean", "avg", "max", "concat", "last"],
)
def test_multiple_dependencies(reduce_dependencies, hidden_shape, dependent_hidden_shape, dependent_hidden_shape2):
    # setup at least for a single dependency
    hidden_layer = torch.randn(hidden_shape, dtype=torch.float32)
    other_hidden_layer = torch.randn(dependent_hidden_shape, dtype=torch.float32)
    other_dependencies = {
        "feature_name": other_hidden_layer,
    }

    # setup dummy output feature to be root of dependency list
    num_feature_defn = number_feature()
    num_feature_defn["loss"] = {"type": "mean_squared_error"}
    num_feature_defn["dependencies"] = ["feature_name"]
    if len(dependent_hidden_shape) > 2:
        num_feature_defn["reduce_dependencies"] = reduce_dependencies

    # Based on specification calculate expected resulting hidden size for
    # with one dependencies
    if reduce_dependencies == "concat" and len(hidden_shape) == 2 and len(dependent_hidden_shape) == 3:
        expected_hidden_size = HIDDEN_SIZE + OTHER_HIDDEN_SIZE * SEQ_SIZE
    else:
        expected_hidden_size = HIDDEN_SIZE + OTHER_HIDDEN_SIZE

    # set up if multiple dependencies specified, setup second dependent feature
    if dependent_hidden_shape2:
        other_hidden_layer2 = torch.randn(dependent_hidden_shape2, dtype=torch.float32)
        other_dependencies["feature_name2"] = other_hidden_layer2
        num_feature_defn["dependencies"].append("feature_name2")
        if len(dependent_hidden_shape2) > 2:
            num_feature_defn["reduce_dependencies"] = reduce_dependencies

        # Based on specification calculate marginal increase in resulting
        # hidden size with two dependencies
        if reduce_dependencies == "concat" and len(hidden_shape) == 2 and len(dependent_hidden_shape2) == 3:
            expected_hidden_size += dependent_hidden_shape2[-1] * SEQ_SIZE
        else:
            expected_hidden_size += dependent_hidden_shape2[-1]

    # Set up dependency reducers.
    dependency_reducers = torch.nn.ModuleDict()
    for feature_name in other_dependencies.keys():
        dependency_reducers[feature_name] = SequenceReducer(reduce_mode=reduce_dependencies)

    # test dependency concatenation
    num_feature_defn["input_size"] = expected_hidden_size
    results = output_feature_utils.concat_dependencies(
        "num_feature", num_feature_defn["dependencies"], dependency_reducers, hidden_layer, other_dependencies
    )

    # confirm size of resulting concat_dependencies() call
    if len(hidden_shape) > 2:
        assert results.shape == (BATCH_SIZE, SEQ_SIZE, expected_hidden_size)
    else:
        assert results.shape == (BATCH_SIZE, expected_hidden_size)


@pytest.mark.parametrize(
    "output_feature_defs",
    [
        generate_output_features_with_dependencies("number_feature", ["category_feature"]),
        generate_output_features_with_dependencies("number_feature", ["category_feature", "sequence_feature"]),
        generate_output_features_with_dependencies("sequence_feature", ["category_feature", "number_feature"]),
    ],
)
def test_construct_output_features_with_dependencies(output_feature_defs):
    # Add keys to output_feature_defs which would have been derived from data.
    def add_data_derived_keys(output_feature_def):
        if DECODER not in output_feature_def:
            output_feature_def[DECODER] = {}
        if output_feature_def[TYPE] == CATEGORY:
            output_feature_def["num_classes"] = 2
        elif output_feature_def[TYPE] == NUMBER:
            output_feature_def[DECODER][TYPE] = "regressor"
        elif output_feature_def[TYPE] == SEQUENCE:
            output_feature_def[DECODER]["max_sequence_length"] = 5
        return output_feature_def

    output_feature_defs = [add_data_derived_keys(of) for of in output_feature_defs]
    # Gets name of output feature which has dependencies.
    dep_feature_name = [of for of in output_feature_defs if len(of.get("dependencies", [])) > 0][0]["name"]
    # Creates a dummy input feature and combiner.
    config = {
        "input_features": [number_feature()],
        "output_features": output_feature_defs,
        "combiner": {"type": "concat", "output_size": 1},
    }
    config_obj = ModelConfig.from_dict(config)
    input_features = BaseModel.build_inputs(config_obj.input_features)
    combiner = ConcatCombiner(input_features=input_features, config=config_obj.combiner)
    output_features = BaseModel.build_outputs(config_obj.output_features, combiner)
    # Gets the output feature object which has dependencies.
    feature_with_deps = output_features[dep_feature_name]
    n_dependencies = len(feature_with_deps.dependencies)
    assert n_dependencies > 0
    # Each synthetic output feature has output size 1, so total size is 1 + n_dependencies.
    assert feature_with_deps.fc_stack.input_shape == torch.Size([1 + n_dependencies])
