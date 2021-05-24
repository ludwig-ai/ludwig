import tensorflow as tf


# taken from https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation.

    Expects 2*n_units-dimensional input.
    Half of it is used to determine the gating of the GLU activation
    and the other half is used as an input to GLU,
    """
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])
