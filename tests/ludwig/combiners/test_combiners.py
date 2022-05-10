import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch

from ludwig.combiners.combiners import (
    ComparatorCombiner,
    ConcatCombiner,
    SequenceCombiner,
    SequenceConcatCombiner,
    TabNetCombiner,
    TabTransformerCombiner,
    TransformerCombiner,
)
from ludwig.encoders.registry import sequence_encoder_registry
from ludwig.schema.combiners import (
    ComparatorCombinerConfig,
    ConcatCombinerConfig,
    SequenceCombinerConfig,
    SequenceConcatCombinerConfig,
    TabNetCombinerConfig,
    TabTransformerCombinerConfig,
    TransformerCombinerConfig,
)
from ludwig.schema.utils import load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SEQ_SIZE = 12
HIDDEN_SIZE = 24
OTHER_HIDDEN_SIZE = 32
OUTPUT_SIZE = 8
BASE_OUTPUT_SIZE = 16
NUM_FILTERS = 20


# emulate Input Feature class.  Need to provide output_shape property to
# mimic what happens during ECD.forward() processing.
class PseudoInputFeature:
    def __init__(self, feature_name, output_shape, feature_type=None):
        self.name = feature_name
        self._output_shape = output_shape
        self.feature_type = feature_type

    def type(self):
        return self.feature_type

    @property
    def output_shape(self):
        return torch.Size(self._output_shape[1:])


# helper function to test correctness of combiner output
def check_combiner_output(combiner, combiner_output, batch_size):
    # check for required attributes
    assert hasattr(combiner, "input_dtype")
    assert hasattr(combiner, "output_shape")

    # check for correct data type
    assert isinstance(combiner_output, dict)

    # required key present
    assert "combiner_output" in combiner_output

    # check for correct output shape
    assert combiner_output["combiner_output"].shape == (batch_size, *combiner.output_shape)


# generates encoder outputs and minimal input feature objects for testing
@pytest.fixture
def features_to_test(feature_list: List[Tuple[str, list]]) -> Tuple[dict, dict]:
    # feature_list: list of tuples that define the output_shape and type
    #    of input features to generate.  tuple[0] is input feature type,
    #    tuple[1] is expected encoder output shape for the input feature
    encoder_outputs = {}
    input_features = {}
    for i in range(len(feature_list)):
        feature_name = f"feature_{i:02d}"
        encoder_outputs[feature_name] = {
            "encoder_output": torch.randn(feature_list[i][1], dtype=torch.float32, device=DEVICE)
        }
        input_features[feature_name] = PseudoInputFeature(feature_name, feature_list[i][1], feature_list[i][0])

    return encoder_outputs, input_features


# set up simulated encoder outputs
@pytest.fixture
def encoder_outputs():
    # generates simulated encoder outputs dictionary:
    #   feature_1: shape [b, h1] tensor
    #   feature_2: shape [b, h2] tensor
    #   feature_3: shape [b, s, h1] tensor
    #   feature_4: shape [b, sh, h2] tensor

    encoder_outputs = {}
    input_features = OrderedDict()
    shapes_list = [
        [BATCH_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, OTHER_HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE],
    ]
    feature_names = ["feature_" + str(i + 1) for i in range(len(shapes_list))]

    for feature_name, batch_shape in zip(feature_names, shapes_list):
        encoder_outputs[feature_name] = {"encoder_output": torch.randn(batch_shape, dtype=torch.float32, device=DEVICE)}
        if len(batch_shape) > 2:
            encoder_outputs[feature_name]["encoder_output_state"] = torch.randn(
                [batch_shape[0], batch_shape[2]], dtype=torch.float32, device=DEVICE
            )

        # create pseudo input feature object
        input_features[feature_name] = PseudoInputFeature(feature_name, batch_shape)

    return encoder_outputs, input_features


# setup encoder outputs for ComparatorCombiner
@pytest.fixture
def encoder_comparator_outputs():
    # generates simulated encoder outputs dictionary:
    #   feature_1: shape [b, h1] tensor
    #   feature_2: shape [b, h2] tensor
    #   feature_3: shape [b, s, h1] tensor
    #   feature_4: shape [b, sh, h2] tensor

    encoder_outputs = {}
    input_features = {}
    shapes_list = [
        [BATCH_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, OTHER_HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE],
    ]
    text_feature_names = ["text_feature_" + str(i + 1) for i in range(len(shapes_list))]
    image_feature_names = ["image_feature_" + str(i + 1) for i in range(len(shapes_list))]
    for i, (feature_name, batch_shape) in enumerate(zip(text_feature_names, shapes_list)):
        # is there a better way to do this?
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_OUTPUT_SIZE]
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(dot_product_shape, dtype=torch.float32, device=DEVICE)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name, dot_product_shape)
        else:
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(batch_shape, dtype=torch.float32, device=DEVICE)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name, batch_shape)

    for i, (feature_name, batch_shape) in enumerate(zip(image_feature_names, shapes_list)):
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_OUTPUT_SIZE]
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(dot_product_shape, dtype=torch.float32, device=DEVICE)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name, dot_product_shape)
        else:
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(batch_shape, dtype=torch.float32, device=DEVICE)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name, batch_shape)

    return encoder_outputs, input_features


# test for simple concatenation combiner
@pytest.mark.parametrize("number_inputs", [None, 1])
@pytest.mark.parametrize("flatten_inputs", [True, False])
@pytest.mark.parametrize("fc_layer", [None, [{"output_size": OUTPUT_SIZE}, {"output_size": OUTPUT_SIZE}]])
def test_concat_combiner(
    encoder_outputs: Tuple, fc_layer: Optional[List[Dict]], flatten_inputs: bool, number_inputs: Optional[int]
) -> None:
    encoder_outputs_dict, input_features_dict = encoder_outputs

    # setup encoder inputs to combiner based on test case
    if not flatten_inputs:
        # clean out rank-3 encoder outputs
        for feature in ["feature_3", "feature_4"]:
            del encoder_outputs_dict[feature]
            del input_features_dict[feature]
        if number_inputs == 1:
            # need only one encoder output for the test
            del encoder_outputs_dict["feature_2"]
            del input_features_dict["feature_2"]
    elif number_inputs == 1:
        # require only one rank-3 encoder output for testing
        for feature in ["feature_1", "feature_2", "feature_3"]:
            del encoder_outputs_dict[feature]
            del input_features_dict[feature]

    # setup combiner to test with pseudo input features
    combiner = ConcatCombiner(
        input_features_dict, config=load_config(ConcatCombinerConfig, fc_layers=fc_layer, flatten_inputs=flatten_inputs)
    ).to(DEVICE)

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]["encoder_output"].shape[1:] == combiner.input_shape[k]

    # combine encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


# test for sequence concatenation combiner
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("main_sequence_feature", [None, "feature_3"])
def test_sequence_concat_combiner(
    encoder_outputs: Tuple, main_sequence_feature: Optional[str], reduce_output: Optional[str]
) -> None:
    # extract encoder outputs and input feature dictionaries
    encoder_outputs_dict, input_feature_dict = encoder_outputs

    # setup combiner for testing
    combiner = SequenceConcatCombiner(
        input_feature_dict,
        config=load_config(
            SequenceConcatCombinerConfig, main_sequence_feature=main_sequence_feature, reduce_output=reduce_output
        ),
    ).to(DEVICE)

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]["encoder_output"].shape[1:] == combiner.input_shape[k]

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += encoder_outputs_dict[k]["encoder_output"].shape[-1]

    # confirm correctness of concatenated_shape
    assert combiner.concatenated_shape[-1] == hidden_size

    # combine encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


# test for sequence combiner
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("encoder", sequence_encoder_registry)
@pytest.mark.parametrize("main_sequence_feature", [None, "feature_3"])
def test_sequence_combiner(
    encoder_outputs: Tuple, main_sequence_feature: Optional[str], encoder: str, reduce_output: Optional[str]
) -> None:
    encoder_outputs_dict, input_features_dict = encoder_outputs

    combiner = SequenceCombiner(
        input_features_dict,
        config=load_config(
            SequenceCombinerConfig,
            main_sequence_feature=main_sequence_feature,
            encoder=encoder,
            reduce_output=reduce_output,
        ),
        # following emulates encoder parameters passed in from config file
        output_size=OUTPUT_SIZE,
        num_fc_layers=3,
    ).to(DEVICE)

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]["encoder_output"].shape[1:] == combiner.input_shape[k]

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += encoder_outputs_dict[k]["encoder_output"].shape[-1]

    # confirm correctness of concatenated_shape
    assert combiner.concatenated_shape[-1] == hidden_size

    # combine encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


@pytest.mark.parametrize(
    "feature_list",  # defines parameter for fixture features_to_test()
    [
        [  # only numeric features
            ("binary", [BATCH_SIZE, 1]),  # passthrough encoder
            ("number", [BATCH_SIZE, 1]),  # passthrough encoder
        ],
        [  # only numeric features
            ("binary", [BATCH_SIZE, 1]),  # passthrough encoder
            ("number", [BATCH_SIZE, 1]),  # passthrough encoder
            ("number", [BATCH_SIZE, 1]),  # passthrough encoder
        ],
        [  # numeric and categorical features
            ("binary", [BATCH_SIZE, 1]),  # passthrough encoder
            ("number", [BATCH_SIZE, 12]),  # dense encoder
            ("category", [BATCH_SIZE, 8]),  # dense encoder
        ],
    ],
)
@pytest.mark.parametrize("size", [4, 8])
@pytest.mark.parametrize("output_size", [6, 10])
def test_tabnet_combiner(features_to_test: Dict, size: int, output_size: int) -> None:
    encoder_outputs, input_features = features_to_test

    # setup combiner to test
    combiner = TabNetCombiner(
        input_features,
        config=load_config(
            TabNetCombinerConfig,
            size=size,
            output_size=output_size,
            num_steps=3,
            num_total_blocks=4,
            num_shared_blocks=2,
            dropout=0.1,
        ),
    ).to(DEVICE)

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs)

    # required key present
    assert "combiner_output" in combiner_output
    assert "attention_masks" in combiner_output
    assert "aggregated_attention_masks" in combiner_output

    assert isinstance(combiner_output["combiner_output"], torch.Tensor)
    assert combiner_output["combiner_output"].shape == (BATCH_SIZE, output_size)


@pytest.mark.parametrize("fc_layer", [None, [{"output_size": 64}, {"output_size": 32}]])
@pytest.mark.parametrize("entity_1", [["text_feature_1", "text_feature_2"]])
@pytest.mark.parametrize("entity_2", [["image_feature_1", "image_feature_2"]])
def test_comparator_combiner(
    encoder_comparator_outputs: Tuple, fc_layer: Optional[List[Dict]], entity_1: str, entity_2: str
) -> None:
    encoder_comparator_outputs_dict, input_features_dict = encoder_comparator_outputs
    # clean out unneeded encoder outputs since we only have 2 layers
    del encoder_comparator_outputs_dict["text_feature_3"]
    del encoder_comparator_outputs_dict["image_feature_3"]
    del encoder_comparator_outputs_dict["text_feature_4"]
    del encoder_comparator_outputs_dict["image_feature_4"]

    # setup combiner to test set to 256 for case when none as it's the default size
    output_size = fc_layer[0]["output_size"] if fc_layer else 256
    combiner = ComparatorCombiner(
        input_features_dict,
        config=load_config(
            ComparatorCombinerConfig, entity_1=entity_1, entity_2=entity_2, fc_layers=fc_layer, output_size=output_size
        ),
    ).to(DEVICE)

    # concatenate encoder outputs
    combiner_output = combiner(encoder_comparator_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


@pytest.mark.parametrize("output_size", [8, 16])
@pytest.mark.parametrize("transformer_output_size", [4, 12])
def test_transformer_combiner(encoder_outputs: tuple, transformer_output_size: int, output_size: int) -> None:
    encoder_outputs_dict, input_feature_dict = encoder_outputs

    # setup combiner to test
    combiner = TransformerCombiner(input_features=input_feature_dict, config=load_config(TransformerCombinerConfig)).to(
        DEVICE
    )

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]["encoder_output"].shape[1:] == combiner.input_shape[k]

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += np.prod(encoder_outputs_dict[k]["encoder_output"].shape[1:])

    # confirm correctness of effective_input_shape
    assert combiner.concatenated_shape[-1] == hidden_size

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


@pytest.mark.parametrize(
    "feature_list",  # defines parameter for fixture features_to_test()
    [
        [  # single numeric, single categorical
            ("number", [BATCH_SIZE, 1]),  # passthrough encoder
            ("category", [BATCH_SIZE, 64]),
        ],
        [  # multiple numeric, multiple categorical
            ("binary", [BATCH_SIZE, 1]),  # passthrough encoder
            ("category", [BATCH_SIZE, 16]),
            ("number", [BATCH_SIZE, 1]),  # passthrough encoder
            ("category", [BATCH_SIZE, 48]),
            ("number", [BATCH_SIZE, 32]),  # dense encoder
        ],
        [  # only numeric features
            ("binary", [BATCH_SIZE, 1]),  # passthrough encoder
            ("number", [BATCH_SIZE, 1]),  # passthrough encoder
        ],
        [("category", [BATCH_SIZE, 16]), ("category", [BATCH_SIZE, 8])],  # only category features
        [("number", [BATCH_SIZE, 1])],  # only single numeric feature  # passthrough encoder
        [("category", [BATCH_SIZE, 8])],  # only single category feature
    ],
)
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("reduce_output", ["concat", "sum"])
@pytest.mark.parametrize("fc_layers", [None, [{"output_size": 256}]])
@pytest.mark.parametrize("embed_input_feature_name", [None, 64, "add"])
def test_tabtransformer_combiner(
    features_to_test: tuple,
    embed_input_feature_name: Optional[Union[int, str]],
    fc_layers: Optional[list],
    reduce_output: str,
    num_layers: int,
) -> None:
    # retrieve simulated encoder outputs and input features for the test
    encoder_outputs, input_features = features_to_test

    # setup combiner to test
    combiner = TabTransformerCombiner(
        input_features=input_features,
        config=load_config(
            TabTransformerCombinerConfig,
            embed_input_feature_name=embed_input_feature_name,
            # emulates parameters passed from combiner def
            num_layers=num_layers,  # number of transformer layers
            fc_layers=fc_layers,  # fully_connected layer definition
            reduce_output=reduce_output,  # sequence reducer
        ),
    ).to(DEVICE)

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs)

    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
