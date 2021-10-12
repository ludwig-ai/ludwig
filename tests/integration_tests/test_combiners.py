import logging
from collections import OrderedDict
import numpy as np
from typing import Optional, Union, List, Tuple
import pytest
import torch

from ludwig.combiners.combiners import (
    ConcatCombiner,
    SequenceConcatCombiner,
    SequenceCombiner,
    TabNetCombiner,
    ComparatorCombiner,
    TransformerCombiner,
    TabTransformerCombiner,
    sequence_encoder_registry,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

BATCH_SIZE = 16
SEQ_SIZE = 12
HIDDEN_SIZE = 24
OTHER_HIDDEN_SIZE = 32
FC_SIZE = 8
BASE_FC_SIZE = 16
NUM_FILTERS = 20


# emulate Input Feature class.  Need to provide output_shape property to
# mimic what happens during ECD.forward() processing.
class PseudoInputFeature:
    def __init__(self, feature_name, output_shape, type=None):
        self.name = feature_name
        self._output_shape = output_shape
        if type is not None:
            self.type = type

    @property
    def output_shape(self):
        return torch.Size(self._output_shape[1:])


# helper function to test correctness of combiner output
def check_combiner_output(combiner, combiner_output, batch_size):
    # check for required attributes
    assert hasattr(combiner, 'input_dtype')
    assert hasattr(combiner, 'output_shape')

    # check for correct data type
    assert isinstance(combiner_output, dict)

    # required key present
    assert 'combiner_output' in combiner_output

    # check for correct output shape
    assert combiner_output['combiner_output'].shape \
           == (batch_size, *combiner.output_shape)


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
        encoder_outputs[feature_name] = {
            "encoder_output": torch.randn(batch_shape, dtype=torch.float32)
        }
        if len(batch_shape) > 2:
            encoder_outputs[feature_name][
                "encoder_output_state"] = torch.randn(
                [batch_shape[0], batch_shape[2]], dtype=torch.float32
            )

        # create pseudo input feature object
        input_features[feature_name] = PseudoInputFeature(feature_name,
                                                          batch_shape)

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
    text_feature_names = ["text_feature_" + str(i + 1) for i in
                          range(len(shapes_list))]
    image_feature_names = [
        "image_feature_" + str(i + 1) for i in range(len(shapes_list))
    ]
    for i, (feature_name, batch_shape) in enumerate(
            zip(text_feature_names, shapes_list)
    ):
        # is there a better way to do this?
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_FC_SIZE]
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(dot_product_shape,
                                              dtype=torch.float32)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name,
                                                              dot_product_shape)
        else:
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(batch_shape,
                                              dtype=torch.float32)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name,
                                                              batch_shape)

    for i, (feature_name, batch_shape) in enumerate(
            zip(image_feature_names, shapes_list)
    ):
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_FC_SIZE]
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(dot_product_shape,
                                              dtype=torch.float32)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name,
                                                              dot_product_shape)
        else:
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(batch_shape,
                                              dtype=torch.float32)
            }
            input_features[feature_name] = PseudoInputFeature(feature_name,
                                                              batch_shape)

    return encoder_outputs, input_features


# test for simple concatenation combiner
@pytest.mark.parametrize("number_inputs", [None, 1])
@pytest.mark.parametrize("flatten_inputs", [True, False])
@pytest.mark.parametrize("fc_layer",
                         [None, [{"fc_size": FC_SIZE}, {"fc_size": FC_SIZE}]])
def test_concat_combiner(encoder_outputs, fc_layer, flatten_inputs,
                         number_inputs):
    encoder_outputs_dict, input_features_dict = encoder_outputs

    # setup encoder inputs to combiner based on test case
    if not flatten_inputs:
        # clean out rank-3 encoder outputs
        for feature in ['feature_3', 'feature_4']:
            del encoder_outputs_dict[feature]
            del input_features_dict[feature]
        if number_inputs == 1:
            # need only one encoder output for the test
            del encoder_outputs_dict['feature_2']
            del input_features_dict['feature_2']
    elif number_inputs == 1:
        # require only one rank-3 encoder output for testing
        for feature in ['feature_1', 'feature_2', 'feature_3']:
            del encoder_outputs_dict[feature]
            del input_features_dict[feature]

    # setup combiner to test with pseudo input features
    combiner = ConcatCombiner(input_features_dict, fc_layers=fc_layer,
                              flatten_inputs=flatten_inputs)

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]['encoder_output'].shape[1:] \
               == combiner.input_shape[k]

    # combine encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


# test for sequence concatenation combiner
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("main_sequence_feature", [None, "feature_3"])
def test_sequence_concat_combiner(
        encoder_outputs, main_sequence_feature, reduce_output
):
    # extract encoder outputs and input feature dictionaries
    encoder_outputs_dict, input_feature_dict = encoder_outputs

    # setup combiner for testing
    combiner = SequenceConcatCombiner(
        input_feature_dict,
        main_sequence_feature=main_sequence_feature,
        reduce_output=reduce_output
    )

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]['encoder_output'].shape[1:] \
               == combiner.input_shape[k]

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
        encoder_outputs, main_sequence_feature, encoder, reduce_output
):
    encoder_outputs_dict, input_features_dict = encoder_outputs

    combiner = SequenceCombiner(
        input_features_dict,
        main_sequence_feature=main_sequence_feature,
        encoder=encoder,
        reduce_output=reduce_output,
        # following emulates encoder parameters passed in from config file
        fc_size=FC_SIZE,
        num_fc_layers=3,
    )

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]['encoder_output'].shape[1:] \
               == combiner.input_shape[k]

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
    'feature_list',  # defines parameter for fixture features_to_test()
    [
        [  # only numeric features
            ('binary', [BATCH_SIZE, 1]),  # passthrough encoder
            ('numerical', [BATCH_SIZE, 1])  # passthrough encoder
        ]
    ]
)
def test_tabnet_combiner(features_to_test):
    encoder_outputs, input_features = features_to_test

    # setup combiner to test
    combiner = TabNetCombiner(
        input_features,
        size=2,
        output_size=2,
        num_steps=3,
        num_total_blocks=4,
        num_shared_blocks=2,
        dropout=0.1
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results
    assert 'attention_masks' in results


@pytest.mark.parametrize("fc_layer",
                         [None, [{"fc_size": 64}, {"fc_size": 32}]])
@pytest.mark.parametrize("entity_1", [["text_feature_1", "text_feature_2"]])
@pytest.mark.parametrize("entity_2", [["image_feature_1", "image_feature_2"]])
def test_comparator_combiner(encoder_comparator_outputs, fc_layer, entity_1,
                             entity_2):
    encoder_comparator_outputs_dict, input_features_dict = encoder_comparator_outputs
    # clean out unneeded encoder outputs since we only have 2 layers
    del encoder_comparator_outputs_dict["text_feature_3"]
    del encoder_comparator_outputs_dict["image_feature_3"]
    del encoder_comparator_outputs_dict["text_feature_4"]
    del encoder_comparator_outputs_dict["image_feature_4"]

    # setup combiner to test set to 256 for case when none as it's the default size
    fc_size = fc_layer[0]["fc_size"] if fc_layer else 256
    combiner = ComparatorCombiner(
        input_features_dict, entity_1, entity_2,
        fc_layers=fc_layer, fc_size=fc_size
    )

    # concatenate encoder outputs
    combiner_output = combiner(encoder_comparator_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


@pytest.mark.parametrize('fc_size', [8, 16])
@pytest.mark.parametrize('transformer_fc_size', [4, 12])
def test_transformer_combiner(
        encoder_outputs: tuple,
        transformer_fc_size: int,
        fc_size: int
) -> None:
    encoder_outputs_dict, input_feature_dict = encoder_outputs

    # setup combiner to test
    combiner = TransformerCombiner(
        input_features=input_feature_dict
    )

    # confirm correctness of input_shape property
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k]['encoder_output'].shape[1:] \
               == combiner.input_shape[k]

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += np.prod(
            encoder_outputs_dict[k]["encoder_output"].shape[1:])

    # confirm correctness of effective_input_shape
    assert combiner.concatenated_shape[-1] == hidden_size

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # check for correctness of combiner output
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)


# generates encoder outputs and minimal input feature objects for testing
@pytest.fixture
def features_to_test(feature_list: List[Tuple[str, list]]) -> Tuple[dict, dict]:
    # feature_list: list of tuples that define the output_shape and type
    #    of input features to generate.  tuple[0] is input feature type,
    #    tuple[1] is expected encoder output shape for the input feature
    encoder_outputs = {}
    input_features = {}
    for i in range(len(feature_list)):
        feature_name = f'feature_{i:02d}'
        encoder_outputs[feature_name] = {
            'encoder_output': torch.randn(feature_list[i][1],
                                          dtype=torch.float32)
        }
        input_features[feature_name] = PseudoInputFeature(
            feature_name,
            feature_list[i][1],
            type=feature_list[i][0]
        )

    return encoder_outputs, input_features


@pytest.mark.parametrize(
    'feature_list',  # defines parameter for fixture features_to_test()
    [
        [  # single numeric, single categorical
            ('numerical', [BATCH_SIZE, 1]),  # passthrough encoder
            ('category', [BATCH_SIZE, 64])
        ],
        [  # multiple numeric, multiple categorical
            ('binary', [BATCH_SIZE, 1]),  # passthrough encoder
            ('category', [BATCH_SIZE, 16]),
            ('numerical', [BATCH_SIZE, 1]),  # passthrough encoder
            ('category', [BATCH_SIZE, 48]),
            ('numerical', [BATCH_SIZE, 32])  # dense encoder
        ],
        [  # only numeric features
            ('binary', [BATCH_SIZE, 1]),  # passthrough encoder
            ('numerical', [BATCH_SIZE, 1])  # passthrough encoder
        ],
        [  # only category features
            ('category', [BATCH_SIZE, 16]),
            ('category', [BATCH_SIZE, 8])
        ],
        [  # only single numeric feature
            ('numerical', [BATCH_SIZE, 1])  # passthrough encoder
        ],
        [  # only single category feature
            ('category', [BATCH_SIZE, 8])
        ]
    ]
)
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('reduce_output', ['concat', 'sum'])
@pytest.mark.parametrize('fc_layers', [None, [{'fc_size': 256}]])
@pytest.mark.parametrize('embed_input_feature_name', [None, 64, 'add'])
def test_tabtransformer_combiner(
        features_to_test: tuple,
        embed_input_feature_name: Optional[Union[int, str]],
        fc_layers: Optional[list],
        reduce_output: str,
        num_layers: int
) -> None:
    # retrieve simulated encoder outputs and input features for the test
    encoder_outputs, input_features = features_to_test

    # setup combiner to test
    combiner = TabTransformerCombiner(
        input_features=input_features,
        embed_input_feature_name=embed_input_feature_name,
        ### emulates parameters passed from combiner def
        num_layers=num_layers,  # number of transformer layers
        fc_layers=fc_layers,  # fully_connected layer definition
        reduce_output=reduce_output  # sequence reducer
    )

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs)

    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
