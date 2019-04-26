import numpy as np
import tensorflow as tf

from ludwig.models.modules.image_encoders import Stacked2DCNN, ResNetEncoder
from ludwig.models.modules.loss_modules import regularizer_registry


L1_REGULARIZER = regularizer_registry['l1'](0.1)
L2_REGULARIZER = regularizer_registry['l2'](0.1)
NO_REGULARIZER = None
DROPOUT_RATE = 0.5


def encoder_test(encoder_type,
                 input_data,
                 regularizer,
                 dropout_rate,
                 output_dtype,
                 output_shape,
                 output_data=None,
                 encoder_args={}):
    """
    Helper method to test different kinds of encoders
    :param encoder_type: type of encoder to test
    :param input_data: data to encode
    :param regularizer: regularizer
    :param dropout_rate: dropout rate
    :param output_dtype: expected data type of the output (optional)
    :param output_shape: expected shape of the encoder output (optional)
    :param output_data: expected output data (optional)
    :param encoder_args: arguments for instantiating the encoder (optional)
    :return: returns the encoder object for the caller to run extra checks
    """

    # Instantiate the encoder
    encoder = encoder_type(**encoder_args)

    # Run the encoder
    input_data = tf.convert_to_tensor(input_data)
    dropout_rate = tf.convert_to_tensor(dropout_rate)
    is_training = tf.convert_to_tensor(False)

    hidden, _ = encoder(input_data, regularizer, dropout_rate,
                        is_training=is_training)

    # Check output shape and type
    assert hidden.dtype == output_dtype
    assert hidden.shape.as_list() == output_shape

    if output_data is not None:
        # TODO the hidden output is actually a tensor. May need modification
        assert np.allclose(hidden, output_data)

    # Return the encoder object so that the caller can run more checks
    # such as verifying the default arguments etc.
    return encoder


def test_image_encoders_resnet():
    # Test the resnet encoder for images
    encoder_args = {'resnet_size': 8, 'num_filters': 8, 'fc_size': 28}
    image_size = (1, 10, 10, 3)

    output_shape = [1, 28]
    input_image = np.random.randint(0, 1, image_size).astype(np.float32)

    encoder = encoder_test(ResNetEncoder, input_image, L1_REGULARIZER,
                           DROPOUT_RATE, np.float, output_shape, None,
                           encoder_args)

    assert encoder is not None
    assert encoder.resnet.kernel_size == 3
