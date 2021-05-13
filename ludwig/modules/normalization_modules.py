import tensorflow as tf

from tensorflow.python.keras.layers import BatchNormalization


class GhostBatchNormalization(tf.keras.Model):
    def __init__(
            self,
            momentum: float = 0.9,
            epsilon: float = 1e-3,
            virtual_batch_size: int = None
    ):
        super().__init__()
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
