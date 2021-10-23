import pytest
from typing import Dict
from copy import deepcopy

import torch

from ludwig.models.ecd import build_single_input
from ludwig.features.image_feature import ImageInputFeature, ENCODER_REGISTRY

BATCH_SIZE = 2


@pytest.fixture(scope='module')
def image_config():
    return {
        'name': 'image_column_name',
        'type': 'image',
        'encoder': 'stacked_cnn',
        'tied_weights': None,
        'conv_layers': None,
        'num_conv_layers': None,
        'filter_size': 3,
        'num_filters': 256,
        'strides': (1, 1),
        'padding': 'valid',
        'dilation_rate': (1, 1),
        'conv_use_bias': True,
        'conv_weights_initializer': 'xavier_uniform',
        'conv_bias_initializer': 'zeros',
        'weights_regularizer': None,
        'conv_bias_regularizer': None,
        'conv_activity_regularizer': None,
        'conv_norm': None,
        'conv_norm_params': None,
        'conv_activation': 'relu',
        'conv_dropout': 0,
        'pool_function': 'max',
        'pool_size': (2, 2),
        'pool_strides': None,
        'fc_layers': None,
        'num_fc_layers': 1,
        'fc_size': 256,
        'fc_use_bias': True,
        'fc_weights_initializer': 'xavier_uniform',
        'fc_bias_initializer': 'zeros',
        'fc_weights_regularizer': None,
        'fc_bias_regularizer': None,
        'fc_activity_regularizer': None,
        'fc_norm': None,
        'fc_norm_params': None,
        'fc_activation': 'relu',
        'fc_dropout': 0,
        'scaling': 'pixel_normalization',
        'preprocessing': {  # example pre-processing
            'height': 28,
            'width': 28,
            'num_channels': 1
        }
    }


@pytest.mark.parametrize(
    'encoder, height, width, num_channels',
    [
        ('resnet', 28, 28, 3),
        ('stacked_cnn', 28, 28, 3),
        ('stacked_cnn', 28, 28, 1),
        ('mlp_mixer', 32, 32, 3),
        ('vit', 224, 224, 3),
    ]
)
def test_image_input_feature(
        image_config: Dict,
        encoder: str,
        height: int,
        width: int,
        num_channels
) -> None:
    # setup image input feature definition
    image_def = deepcopy(image_config)
    image_def['encoder'] = encoder
    image_def['height'] = height
    image_def['width'] = width
    image_def['num_channels'] = num_channels

    # pickup any other missing parameters
    ImageInputFeature.populate_defaults(image_def)

    # ensure no exceptions raised during build
    input_feature_obj = build_single_input(image_def, None)

    # check one forward pass through input feature
    input_tensor = torch.randint(0, 256,
                                 size=(BATCH_SIZE, num_channels, height, width),
                                 dtype=torch.uint8)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output['encoder_output'].shape == \
           (BATCH_SIZE, *input_feature_obj.output_shape)

    # todo: remove code
    # # test for parameter updates
    # before = [(x[0], x[1].clone()) for x in input_feature_obj.named_parameters()]
    # loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(input_feature_obj.parameters(), lr=0.1)
    # target_tensor = torch.ones(encoder_output['encoder_output'].shape, dtype=torch.float32)
    #
    # # do parameter update
    # loss = loss_function(encoder_output['encoder_output'], target_tensor)
    # loss.backward()
    # optimizer.step()
    #
    # after = [(x[0], x[1].clone()) for x in input_feature_obj.named_parameters()]
    #
    # # check for parameter update
    # for b, a in zip(before, after):
    #     if not (b[1] != a[1]).any():
    #         raise RuntimeError(
    #             f'no parameter update for {a[0]}'
    #         )
