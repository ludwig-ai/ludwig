import logging
from collections import OrderedDict

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


# emulate Input Feature class.  Need to provide output_shape property to
# mimic what happens during ECD.forward() processing.
class PseudoInputFeature:
    def __init__(self, feature_name, output_shape):
        self.name = feature_name
        self._output_shape = output_shape

    @property
    def output_shape(self):
        return torch.Size(self._output_shape[1:])


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
        else:
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(batch_shape,
                                              dtype=torch.float32)
            }

    for i, (feature_name, batch_shape) in enumerate(
            zip(image_feature_names, shapes_list)
    ):
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_FC_SIZE]
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(dot_product_shape,
                                              dtype=torch.float32)
            }
        else:
            encoder_outputs[feature_name] = {
                "encoder_output": torch.randn(batch_shape,
                                              dtype=torch.float32)
            }

    return encoder_outputs


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

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # correct data structure
    assert isinstance(combiner_output, dict)

    # required key present and correct data type
    assert "combiner_output" in combiner_output
    assert isinstance(combiner_output['combiner_output'], torch.Tensor)

    # confirm correct output shapes
    if fc_layer:
        assert combiner_output["combiner_output"].shape == (BATCH_SIZE, FC_SIZE)
    else:
        # calculate expected hidden size for concatenated tensors
        hidden_size = 0
        for k in encoder_outputs_dict:
            hidden_size += encoder_outputs_dict[k]["encoder_output"].shape[1] \
                if not flatten_inputs else \
                encoder_outputs_dict[k]["encoder_output"] \
                    .reshape(BATCH_SIZE, -1).shape[1]

        assert combiner_output["combiner_output"].shape == \
               (BATCH_SIZE, hidden_size)


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

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += encoder_outputs_dict[k]["encoder_output"].shape[-1]

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # correct data structure
    assert isinstance(combiner_output, dict)

    # required key present and correct data type
    assert "combiner_output" in combiner_output
    assert isinstance(combiner_output['combiner_output'], torch.Tensor)

    # confirm correct shape
    if reduce_output is None:
        assert combiner_output["combiner_output"].shape == (
            BATCH_SIZE,
            SEQ_SIZE,
            hidden_size,
        )
    else:
        assert combiner_output["combiner_output"].shape == (
        BATCH_SIZE, hidden_size)


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
    )

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += encoder_outputs_dict[k]["encoder_output"].shape[-1]

    # concatenate encoder outputs
    combiner_output = combiner(encoder_outputs_dict)

    # correct data structure
    assert isinstance(combiner_output, dict)

    # required key present and correct data type
    assert "combiner_output" in combiner_output
    assert isinstance(combiner_output['combiner_output'], torch.Tensor)

    combiner_shape = combiner_output["combiner_output"].shape
    # test for correct dimension
    if reduce_output:
        assert len(combiner_shape) == 2
    else:
        assert len(combiner_shape) == 3

    # Shape test assumes on Ludwig sequence encoder defaults
    #   parallel encoders: # layers = 4, fc_size=256
    #   non-parallel encoders: fc_size=256
    # if defaults change, then this test has to be updated
    default_layer = 4
    default_fc_size = 256

    if "parallel" in encoder:
        assert combiner_shape[-1] == default_layer * default_fc_size
    else:
        assert combiner_shape[-1] == default_fc_size


def tabnet_encoder_outputs():
    # Need to do this in a function, otherwise TF will try to initialize
    # too early
    return {
        'batch_128': {
            'feature_1': {
                'encoder_output': torch.randn(
                    [128, 1],
                    dtype=torch.float32
                )
            },
            'feature_2': {
                'encoder_output': torch.randn(
                    [128, 1],
                    dtype=torch.float32
                )
            },
        },
        'inputs': {
            'feature_1': {
                'encoder_output': tf.keras.Input(
                    (),
                    dtype=torch.float32,
                    name='feature_1',
                )
            },
            'feature_2': {
                'encoder_output': tf.keras.Input(
                    (),
                    dtype=torch.float32,
                    name='feature_2',
                )
            },
        }
    }


@pytest.mark.parametrize("encoder_outputs_key", ['batch_128', 'inputs'])
def test_tabnet_combiner(encoder_outputs_key):
    encoder_outputs = tabnet_encoder_outputs()[encoder_outputs_key]

    # setup combiner to test
    combiner = TabNetCombiner(
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
                         [None, [{"fc_size": 64}, {"fc_size": 64}]])
@pytest.mark.parametrize("entity_1", [["text_feature_1", "text_feature_2"]])
@pytest.mark.parametrize("entity_2", [["image_feature_1", "image_feature_2"]])
def test_comparator_combiner(encoder_comparator_outputs, fc_layer, entity_1,
                             entity_2):
    # clean out unneeded encoder outputs since we only have 2 layers
    del encoder_comparator_outputs["text_feature_3"]
    del encoder_comparator_outputs["image_feature_3"]
    del encoder_comparator_outputs["text_feature_4"]
    del encoder_comparator_outputs["image_feature_4"]

    # setup combiner to test set to 256 for case when none as it's the default size
    fc_size = fc_layer[0]["fc_size"] if fc_layer else 256
    combiner = ComparatorCombiner(
        entity_1, entity_2, fc_layers=fc_layer, fc_size=fc_size
    )

    # concatenate encoder outputs
    results = combiner(encoder_comparator_outputs)

    # required key present
    assert "combiner_output" in results

    # confirm correct output shapes
    # concat on axis=1
    # because of dot products, 2 of the shapes added will be the fc_size
    #   other 2 will be of shape BATCH_SIZE
    # this assumes dimensionality = 2
    size = BATCH_SIZE * 2 + fc_size * 2
    assert results["combiner_output"].shape.as_list() == [BATCH_SIZE, size]


def test_transformer_combiner(encoder_outputs):
    # clean out unneeded encoder outputs
    encoder_outputs = {}
    encoder_outputs['feature_1'] = {
        'encoder_output': torch.randn(
            [128, 1],
            dtype=torch.float32
        )
    }
    encoder_outputs['feature_2'] = {
        'encoder_output': torch.randn(
            [128, 1],
            dtype=torch.float32
        )
    }

    input_features_def = [
        {'name': 'feature_1', 'type': 'numerical'},
        {'name': 'feature_2', 'type': 'numerical'}
    ]

    # setup combiner to test
    combiner = TransformerCombiner(
        input_features=input_features_def
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results


def test_tabtransformer_combiner(encoder_outputs):
    # clean out unneeded encoder outputs
    encoder_outputs = {}
    encoder_outputs['feature_1'] = {
        'encoder_output': torch.randn(
            [128, 1],
            dtype=torch.float32
        )
    }
    encoder_outputs['feature_2'] = {
        'encoder_output': torch.randn(
            [128, 16],
            dtype=torch.float32
        )
    }

    input_features_def = [
        {'name': 'feature_1', 'type': 'numerical'},
        {'name': 'feature_2', 'type': 'category'}
    ]

    # setup combiner to test
    combiner = TabTransformerCombiner(
        input_features=input_features_def
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results

    # setup combiner to test
    combiner = TabTransformerCombiner(
        input_features=input_features_def,
        embed_input_feature_name=56
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results

    # setup combiner to test
    combiner = TabTransformerCombiner(
        input_features=input_features_def,
        embed_input_feature_name='add'
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results
