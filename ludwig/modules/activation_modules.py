import tensorflow as tf


# taken from https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation.

    Expects 2*n_units-dimensional input.
    Half of it is used to determine the gating of the GLU activation
    and the other half is used as an input to GLU,
    """
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])


def gelu(features, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, features.dtype)
        return 0.5 * features * (
                1.0 + tf.math.tanh(0.7978845608028654 *
                                   (features + coeff * tf.math.pow(features,
                                                                   3))))
    else:
        return 0.5 * features * (1.0 + tf.math.erf(
            features / tf.cast(1.4142135623730951, features.dtype)))
