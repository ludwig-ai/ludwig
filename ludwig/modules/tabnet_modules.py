from typing import List, Tuple

import tensorflow as tf
from tensorflow_addons.layers import Sparsemax

from ludwig.modules.activation_modules import glu
from ludwig.modules.normalization_modules import GhostBatchNormalization


# from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/model.py
class TabNet(tf.keras.Model):
    def __init__(
            self,
            num_features: int,
            size: int,
            output_size: int,
            num_steps: int = 1,
            num_total_blocks: int = 4,
            num_shared_blocks: int = 2,
            relaxation_factor: float = 1.5,
            bn_epsilon: float = 1e-5,
            bn_momentum: float = 0.7,
            bn_virtual_divider: int = 1,
            sparsity: float = 1e-5,
    ):
        """TabNet
        Will output a vector of size output_dim.
        Args:
            num_features (int): Number of features.
            size (int): Embedding feature dimension to use.
            output_size (int): Output dimension.
            num_steps (int, optional): Total number of steps. Defaults to 1.
            num_total_blocks (int, optional): Total number of feature transformer blocks. Defaults to 4.
            num_shared_blocks (int, optional): Number of shared feature transformer blocks. Defaults to 2.
            relaxation_factor (float, optional): >1 will allow features to be used more than once. Defaults to 1.5.
            bn_epsilon (float, optional): Batch normalization, epsilon. Defaults to 1e-5.
            bn_momentum (float, optional): Batch normalization, momentum. Defaults to 0.7.
            bn_virtual_divider (int, optional): Batch normalization. Full batch will be divided by this.
        """
        super(TabNet, self).__init__()
        self.num_features = num_features
        self.size = size
        self.output_size = output_size
        self.num_steps = num_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity = sparsity

        # ? Switch to Ghost Batch Normalization
        self.batch_norm = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "size": size + output_size,
            "num_total_blocks": num_total_blocks,
            "num_shared_blocks": num_shared_blocks,
            "bn_momentum": bn_momentum,
            "bn_virtual_divider": bn_virtual_divider,
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms: List[FeatureTransformer] = [
            FeatureTransformer(**kargs)
        ]
        self.attentive_transforms: List[AttentiveTransformer] = [None, ]
        for i in range(num_steps):
            self.feature_transforms.append(
                FeatureTransformer(**kargs,
                                   shared_fc_layers=self.feature_transforms[
                                       0].shared_fc_layers)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features, bn_momentum,
                                     bn_virtual_divider)
            )
        self.final_projection = tf.keras.layers.Dense(self.output_size)


    def call(
            self,
            features: tf.Tensor,
            training: bool = None,
            alpha: float = 0.0
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        batch_size = tf.shape(features)[0]
        out_accumulator = tf.zeros((batch_size, self.size))
        prior_scales = tf.ones((batch_size, self.num_features))
        masks = []
        total_entropy = 0.0

        features = self.batch_norm(features, training=training)
        masked_features = features

        x = self.feature_transforms[0](
            masked_features, training=training, alpha=alpha
        )

        for step_i in range(1, self.num_steps + 1):
            ########################
            # Attentive Transormer #
            ########################
            mask_values = self.attentive_transforms[step_i](
                x[:, self.output_size :], prior_scales, training=training, alpha=alpha
            )

            # relaxation factor 1 forces the feature to be only used once
            prior_scales *= self.relaxation_factor - mask_values

            # entropy is used to penalize the amount of sparsity
            # in feature selection
            total_entropy = tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(mask_values,
                                tf.math.log(mask_values + 1e-15)),
                    axis=1,
                )
            )

            masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))

            ########################
            # Feature Transormer #
            ########################
            masked_features = tf.multiply(mask_values, features)

            x = self.feature_transforms[step_i](
                masked_features, training=training, alpha=alpha
            )

            out = tf.keras.activations.relu(x[:, : self.output_size])
            out_accumulator += out

        final_output = self.final_projection(out_accumulator)

        self.add_loss(-self.sparsity * total_entropy / self.num_steps)

        return final_output, masks


# from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/transformers.py
class FeatureBlock(tf.keras.Model):
    def __init__(
            self,
            size: int,
            apply_glu: bool = True,
            bn_momentum: float = 0.9,
            bn_virtual_divider: int = 32,
            shared_fc_layer: tf.keras.layers.Layer = None,
            epsilon: float = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_glu = apply_glu
        self.size = size
        units = size * 2 if apply_glu else size

        if shared_fc_layer:
            self.fc_layer = shared_fc_layer
        else:
            self.fc_layer = tf.keras.layers.Dense(units, use_bias=False)

        self.batch_norm = GhostBatchNormalization(
            virtual_divider=bn_virtual_divider, momentum=bn_momentum
        )

    def call(self, inputs, training: bool = None, alpha: float = 0.0):
        hidden = self.fc_layer(inputs)
        hidden = self.batch_norm(hidden, training=training, alpha=alpha)
        if self.apply_glu:
            hidden = glu(hidden, self.size)
        return hidden


# from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/transformers.py
class AttentiveTransformer(tf.keras.Model):
    def __init__(
            self,
            size: int,
            bn_momentum: float = 0.9,
            bn_virtual_divider: int = 32,
    ):
        super(AttentiveTransformer, self).__init__()
        self.feature_block = FeatureBlock(
            size,
            bn_momentum=bn_momentum,
            bn_virtual_divider=bn_virtual_divider,
            apply_glu=False,
        )
        self.sparsemax = Sparsemax()

    def call(self, inputs, prior_scales, training=None, alpha: float = 0.0):
        hidden = self.feature_block(inputs, training=training, alpha=alpha)
        return self.sparsemax(hidden * prior_scales)


# from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/transformers.py
class FeatureTransformer(tf.keras.Model):
    def __init__(
            self,
            size: int,
            shared_fc_layers: List[tf.keras.layers.Layer] = [],
            num_total_blocks: int = 4,
            num_shared_blocks: int = 2,
            bn_momentum: float = 0.9,
            bn_virtual_divider: int = 1,
    ):
        super(FeatureTransformer, self).__init__()
        self.num_total_blocks = num_total_blocks
        self.num_shared_blocks = num_shared_blocks

        kargs = {
            "size": size,
            "bn_momentum": bn_momentum,
            "bn_virtual_divider": bn_virtual_divider,
        }

        # build blocks
        self.blocks: List[FeatureBlock] = []
        for n in range(num_total_blocks):
            if shared_fc_layers and n < len(shared_fc_layers):
                # add shared blocks
                self.blocks.append(
                    FeatureBlock(**kargs, shared_fc_layer=shared_fc_layers[n]))
            else:
                # build new blocks
                self.blocks.append(FeatureBlock(**kargs))

    def call(
            self,
            inputs: tf.Tensor,
            training: bool = None,
            alpha: float = 0.0
    ) -> tf.Tensor:
        hidden = self.blocks[0](inputs, training=training, alpha=alpha)
        for n in range(1, self.num_total_blocks):
            hidden = (hidden * tf.sqrt(0.5) +
                      self.blocks[n](hidden, training=training, alpha=alpha))
        return hidden

    @property
    def shared_fc_layers(self):
        return [self.blocks[i].fc_layer for i in range(self.num_shared_blocks)]
