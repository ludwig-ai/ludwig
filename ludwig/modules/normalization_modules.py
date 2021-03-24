import tensorflow as tf


# from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/gbn.py
class GhostBatchNormalization(tf.keras.Model):
    def __init__(
            self, virtual_divider: int = 1, momentum: float = 0.9,
            epsilon: float = 1e-5
    ):
        super(GhostBatchNormalization, self).__init__()
        self.virtual_divider = virtual_divider
        self.bn = BatchNormInferenceWeighting(momentum=momentum)

    def call(self, x, training: bool = None, alpha: float = 0.0):
        if training:
            num_or_size_splits = self.virtual_divider
            if x.shape[0] % self.virtual_divider != 0:
                q, r = divmod(x.shape[0], self.virtual_divider)
                num_or_size_splits = [q] * self.virtual_divider + [r]
            chunks = tf.split(x, num_or_size_splits)
            x = [self.bn(x, training=True) for x in chunks]
            return tf.concat(x, 0)
        return self.bn(x, training=False, alpha=alpha)

    @property
    def moving_mean(self):
        return self.bn.moving_mean

    @property
    def moving_variance(self):
        return self.bn.moving_variance


# from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/gbn.py
class BatchNormInferenceWeighting(tf.keras.layers.Layer):
    def __init__(self, momentum: float = 0.9, epsilon: float = None):
        super(BatchNormInferenceWeighting, self).__init__()
        self.momentum = momentum
        self.epsilon = tf.keras.backend.epsilon() if epsilon is None else epsilon

    def build(self, input_shape):
        channels = input_shape[-1]

        self.gamma = tf.Variable(
            initial_value=tf.ones((channels,), tf.float32), trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32), trainable=True,
        )

        self.moving_mean = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32), trainable=False,
        )
        self.moving_mean_of_squares = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32), trainable=False,
        )

    def __update_moving(self, var, value):
        var.assign(var * self.momentum + (1 - self.momentum) * value)

    def __apply_normalization(self, x, mean, variance):
        return self.gamma * (x - mean) / tf.sqrt(
            variance + self.epsilon) + self.beta

    def call(self, x, training: bool = None, alpha: float = 0.0):
        mean = tf.reduce_mean(x, axis=0)
        mean_of_squares = tf.reduce_mean(tf.pow(x, 2), axis=0)

        if training:
            # update moving stats
            self.__update_moving(self.moving_mean, mean)
            self.__update_moving(self.moving_mean_of_squares, mean_of_squares)

            variance = mean_of_squares - tf.pow(mean, 2)
            x = self.__apply_normalization(x, mean, variance)
        else:
            mean = alpha * mean + (1 - alpha) * self.moving_mean
            variance = (alpha * mean_of_squares + (1 - alpha) *
                        self.moving_mean_of_squares) - tf.pow(mean, 2)
            x = self.__apply_normalization(x, mean, variance)

        return x
