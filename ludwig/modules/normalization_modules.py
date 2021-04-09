import tensorflow as tf

from tensorflow.python.keras.layers import BatchNormalization


class GhostBatchNormalization(tf.keras.Model):
    def __init__(
            self,
            momentum: float = 0.9,
            epsilon: float = 1e-5,
            virtual_batch_size: int = None
    ):
        super(GhostBatchNormalization, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, x, training: bool = None, **kwargs):
        if training and self.virtual_batch_size:
            batch_size = x.shape[0]

            q, r = divmod(batch_size, self.virtual_batch_size)
            num_or_size_splits = q
            if r != 0:
                num_or_size_splits = [self.virtual_batch_size] * q + [r]

            splits = tf.split(x, num_or_size_splits)
            x = [self.bn(x, training=True) for x in splits]
            return tf.concat(x, 0)

        return self.bn(x, training=False)

    @property
    def moving_mean(self):
        return self.bn.moving_mean

    @property
    def moving_variance(self):
        return self.bn.moving_variance

# adapted and modified from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/gbn.py
# class BatchNormInferenceWeighting(tf.keras.layers.Layer):
#     def __init__(self, momentum: float = 0.9, epsilon: float = None):
#         super(BatchNormInferenceWeighting, self).__init__()
#         self.momentum = momentum
#         self.epsilon = tf.keras.backend.epsilon() if epsilon is None else epsilon
#
#     def build(self, input_shape):
#         channels = input_shape[-1]
#
#         self.gamma = tf.Variable(
#             initial_value=tf.ones((channels,), tf.float32), trainable=True,
#             name="gamma"
#         )
#         self.beta = tf.Variable(
#             initial_value=tf.zeros((channels,), tf.float32), trainable=True,
#             name="beta"
#         )
#
#         self.moving_mean = tf.Variable(
#             initial_value=tf.zeros((channels,), tf.float32), trainable=False,
#             name="moving_mean"
#         )
#         self.moving_mean_of_squares = tf.Variable(
#             initial_value=tf.zeros((channels,), tf.float32), trainable=False,
#             name="moving_mean_of_squares"
#         )
#
#     def update_moving(self, var, value):
#         var.assign(var * self.momentum + (1 - self.momentum) * value)
#
#     def apply_normalization(self, x, mean, variance):
#         return self.gamma * (x - mean) / tf.sqrt(
#             variance + self.epsilon) + self.beta
#
#     def call(self, x, training: bool = None, alpha: float = 0.0):
#         mean = tf.reduce_mean(x, axis=0)
#         mean_of_squares = tf.reduce_mean(tf.pow(x, 2), axis=0)
#
#         if training:
#             # update moving stats
#             self.update_moving(self.moving_mean, mean)
#             self.update_moving(self.moving_mean_of_squares, mean_of_squares)
#
#             variance = mean_of_squares - tf.pow(mean, 2)
#             x = self.apply_normalization(x, mean, variance)
#         else:
#             mean = alpha * mean + (1 - alpha) * self.moving_mean
#             variance = (alpha * mean_of_squares + (1 - alpha) *
#                         self.moving_mean_of_squares) - tf.pow(mean, 2)
#             x = self.apply_normalization(x, mean, variance)
#
#         return x
