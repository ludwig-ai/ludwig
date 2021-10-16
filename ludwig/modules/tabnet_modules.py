from typing import List, Tuple, Optional

import torch
from torch.autograd import Function

from ludwig.modules.activation_modules import glu
from ludwig.modules.normalization_modules import GhostBatchNormalization
from ludwig.utils.torch_utils import LudwigModule


class TabNet(LudwigModule):
    def __init__(
            self,
            input_size: int,
            size: int,
            output_size: int,
            num_steps: int = 1,
            num_total_blocks: int = 4,
            num_shared_blocks: int = 2,
            relaxation_factor: float = 1.5,
            bn_momentum: float = 0.7,
            bn_epsilon: float = 1e-3,
            bn_virtual_bs: int = None,
            sparsity: float = 1e-5,
    ):
        """TabNet
        Will output a vector of size output_dim.
        Args:
            input_size (int): concatenated size of input feature encoder outputs
            size (int): Embedding feature dimension
            output_size (int): Output dimension for TabNet
            num_steps (int, optional): Total number of steps. Defaults to 1.
            num_total_blocks (int, optional): Total number of feature transformer blocks. Defaults to 4.
            num_shared_blocks (int, optional): Number of shared feature transformer blocks. Defaults to 2.
            relaxation_factor (float, optional): >1 will allow features to be used more than once. Defaults to 1.5.
            bn_momentum (float, optional): Batch normalization, momentum. Defaults to 0.7.
            bn_epsilon (float, optional): Batch normalization, epsilon. Defaults to 1e-5.
            bn_virtual_bs (int, optional): Virtual batch ize for ghost batch norm..
        """
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.output_size = output_size
        self.num_steps = num_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity = torch.tensor(sparsity)

        # needed by the attentive transformer in build()
        self.num_steps = num_steps
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.bn_virtual_bs = bn_virtual_bs

        self.batch_norm = torch.nn.BatchNorm1d(input_size,
                                               momentum=bn_momentum,
                                               eps=bn_epsilon
                                               )

        kargs = {
            "size": size + output_size,
            "num_total_blocks": num_total_blocks,
            "num_shared_blocks": num_shared_blocks,
            "bn_momentum": bn_momentum,
            "bn_epsilon": bn_epsilon,
            "bn_virtual_bs": bn_virtual_bs,
        }

        # first feature transformer block is built first
        # to get the shared blocks
        self.feature_transforms = torch.nn.ModuleList([
            FeatureTransformer(input_size, **kargs)
        ])
        self.attentive_transforms = torch.nn.ModuleList()
        for i in range(num_steps):
            self.feature_transforms.append(
                FeatureTransformer(
                    input_size,
                    **kargs,
                    shared_fc_layers=self.feature_transforms[
                        0].shared_fc_layers
                )
            )
            # attentive transformers are initialized in build
            # because their outputs size depends on the number
            # of features that we determine by looking at the
            # last dimension of the input tensor
            self.attentive_transforms.append(
                AttentiveTransformer(size, input_size, bn_momentum,
                                     bn_epsilon, bn_virtual_bs)
            )
        self.final_projection = torch.nn.Linear(
            output_size,
            output_size)

    def forward(
            self,
            features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if features.dim() != 2:
            raise ValueError(
                f'Expecting incoming tensor to be dim 2, '
                f'instead dim={features.dim()}'
            )

        batch_size = features.shape[0]
        num_features = features.shape[-1]
        out_accumulator = torch.zeros((batch_size, self.output_size))
        aggregated_mask = torch.zeros([batch_size, num_features])
        prior_scales = torch.ones((batch_size, num_features))
        masks = []
        total_entropy = 0.0

        features = self.batch_norm(features)
        masked_features = features

        x = self.feature_transforms[0](masked_features)

        for step_i in range(1, self.num_steps):  # todo: old value num_steps + 1
            #########################
            # Attentive Transformer #
            #########################
            mask_values = self.attentive_transforms[step_i](
                x[:, self.output_size:], prior_scales)

            # relaxation factor 1 forces the feature to be only used once
            prior_scales *= self.relaxation_factor - mask_values

            # entropy is used to penalize the amount of sparsity
            # in feature selection
            total_entropy += torch.mean(
                torch.sum(
                    -mask_values * torch.log(mask_values + 0.00001),
                    dim=1)
            ) / self.num_steps

            masks.append(torch.unsqueeze(torch.unsqueeze(mask_values, 0), 3))

            #######################
            # Feature Transformer #
            #######################
            masked_features = torch.multiply(mask_values, features)

            x = self.feature_transforms[step_i](
                masked_features
            )

            out = torch.nn.functional.relu(x[:, :self.output_size])
            out_accumulator += out

            # Aggregated masks are used for visualization of the
            # feature importance attributes.
            scale = torch.sum(out, dim=1, keepdim=True) / self.num_steps
            aggregated_mask += mask_values * scale

        final_output = self.final_projection(out_accumulator)

        sparsity_loss = torch.multiply(self.sparsity, total_entropy)
        setattr(sparsity_loss, "loss_name", "sparsity_loss")
        self.add_loss(sparsity_loss)

        return final_output, aggregated_mask, masks


class FeatureBlock(LudwigModule):
    def __init__(
            self,
            input_size: int,
            size: int,
            apply_glu: bool = True,
            bn_momentum: float = 0.9,
            bn_epsilon: float = 1e-3,
            bn_virtual_bs: int = None,
            shared_fc_layer: LudwigModule = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.apply_glu = apply_glu
        self.size = size
        units = size * 2 if apply_glu else size

        if shared_fc_layer:
            self.fc_layer = shared_fc_layer
        else:
            self.fc_layer = torch.nn.Linear(input_size, units, bias=False)

        self.batch_norm = GhostBatchNormalization(
            units,
            virtual_batch_size=bn_virtual_bs,
            momentum=bn_momentum,
            epsilon=bn_epsilon
        )

    def forward(self, inputs):
        hidden = self.fc_layer(inputs)
        hidden = self.batch_norm(hidden)
        if self.apply_glu:
            hidden = glu(hidden)
        return hidden

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])


class AttentiveTransformer(LudwigModule):
    def __init__(
            self,
            input_size: int,
            size: int,
            bn_momentum: float = 0.9,
            bn_epsilon: float = 1e-3,
            bn_virtual_bs: int = None,
    ):
        super().__init__()
        self.input_size = input_size

        self.feature_block = FeatureBlock(
            input_size,
            size,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            bn_virtual_bs=bn_virtual_bs,
            apply_glu=False,
        )
        self.sparsemax = Sparsemax()
        # self.sparsemax = CustomSparsemax()  # todo: tf implementation

    def forward(self, inputs, prior_scales):
        hidden = self.feature_block(inputs)
        hidden = hidden * prior_scales

        # removing the mean to try to avoid numerical instability
        # https://github.com/tensorflow/addons/issues/2314
        # https://github.com/tensorflow/tensorflow/pull/21183/files
        # In the paper, they call the logits z.
        # The mean(logits) can be substracted from logits to make the algorithm
        # more numerically stable. the instability in this algorithm comes mostly
        # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
        # to zero.
        # hidden = hidden - tf.math.reduce_mean(hidden, axis=1)[:, tf.newaxis]

        # added to avoid NaNs in the sparsemax
        hidden = torch.clamp(hidden, min=-1.0e+6, max=1.0e+6)
        return self.sparsemax(hidden)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        tensor = torch.rand([2, *self.input_shape], dtype=torch.float32)
        prior_scales = torch.ones(2, self.input_shape[-1])
        output = self.forward(tensor, prior_scales)
        return output.size()[1:]

# adapted and modified from https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/transformers.py
class FeatureTransformer(LudwigModule):
    def __init__(
            self,
            input_size: int,
            size: int,
            shared_fc_layers: List = [],
            num_total_blocks: int = 4,
            num_shared_blocks: int = 2,
            bn_momentum: float = 0.9,
            bn_epsilon: float = 1e-3,
            bn_virtual_bs: int = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_total_blocks = num_total_blocks
        self.num_shared_blocks = num_shared_blocks
        self.size = size

        kwargs = {
            "bn_momentum": bn_momentum,
            "bn_epsilon": bn_epsilon,
            "bn_virtual_bs": bn_virtual_bs,
        }

        # build blocks
        self.blocks = torch.nn.ModuleList()
        for n in range(num_total_blocks):
            if shared_fc_layers and n < len(shared_fc_layers):
                # add shared blocks
                self.blocks.append(
                    FeatureBlock(
                        input_size,
                        size,
                        **kwargs,
                        shared_fc_layer=shared_fc_layers[n]
                    )
                )
            else:
                # build new blocks
                if n == 0:
                    # first block
                    self.blocks.append(FeatureBlock(input_size, size, **kwargs))
                else:
                    # subsequent blocks
                    self.blocks.append(FeatureBlock(size, size, **kwargs))

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.blocks[0](inputs)
        for n in range(1, self.num_total_blocks):
            hidden = (self.blocks[n](hidden) +
                      hidden) * torch.sqrt(torch.tensor(0.5))
        return hidden

    @property
    def shared_fc_layers(self):
        return [self.blocks[i].fc_layer for i in range(self.num_shared_blocks)]

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.size])


# sparsemax implementation: https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/sparsemax.py
# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax
        Returns
        -------
        output : torch.Tensor
            same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input,
                                                                  dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax
        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(torch.nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)

# todo: clean up #############################
# reimplementation of sparsemax to be more stable and fallback to softmax
# adapted from https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/activations/sparsemax.py#L21-L77
# class CustomSparsemax(LudwigModule):
#     """Sparsemax activation function.
#
#     The output shape is the same as the input shape.
#
#     See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
#
#     Args:
#         axis: Integer, axis along which the sparsemax normalization is applied.
#     """
#
#     def __init__(self, axis: int = -1, **kwargs):
#         super().__init__(**kwargs)
#         self.supports_masking = True
#         self.axis = axis
#
#     def call(self, inputs, **kwargs):
#         return sparsemax(inputs, axis=self.axis)
#
#     def get_config(self):
#         config = {"axis": self.axis}
#         base_config = super().get_config()
#         return {**base_config, **config}
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#
# def sparsemax(logits, axis: int = -1) -> torch.Tensor:
#     r"""Sparsemax activation function.
#
#     For each batch $i$, and class $j$,
#     compute sparsemax activation function:
#
#     $$
#     \mathrm{sparsemax}(x)[i, j] = \max(\mathrm{logits}[i, j] - \tau(\mathrm{logits}[i, :]), 0).
#     $$
#
#     See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
#
#     Usage:
#
#     x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
#     tfa.activations.sparsemax(x)
#     <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
#     array([[0., 0., 1.],
#            [0., 0., 1.]], dtype=float32)>
#
#     Args:
#         logits: A `Tensor`.
#         axis: `int`, axis along which the sparsemax operation is applied.
#     Returns:
#         A `Tensor`, output of sparsemax transformation. Has the same type and
#         shape as `logits`.
#     Raises:
#         ValueError: In case `dim(logits) == 1`.
#     """
#     logits = tf.convert_to_tensor(logits, name="logits")
#
#     # We need its original shape for shape inference.
#     shape = logits.get_shape()
#     rank = shape.rank
#     is_last_axis = (axis == -1) or (axis == rank - 1)
#
#     if is_last_axis:
#         output = _compute_2d_sparsemax(logits)
#         output.set_shape(shape)
#         return output
#
#     # If dim is not the last dimension, we have to do a transpose so that we can
#     # still perform softmax on its last dimension.
#
#     # Swap logits' dimension of dim and its last dimension.
#     rank_op = tf.rank(logits)
#     axis_norm = axis % rank
#     logits = _swap_axis(logits, axis_norm,
#                         tf.math.subtract(rank_op, 1))
#
#     # Do the actual softmax on its last dimension.
#     output = _compute_2d_sparsemax(logits)
#     output = _swap_axis(output, axis_norm,
#                         tf.math.subtract(rank_op, 1))
#
#     # Make shape inference work since transpose may erase its static shape.
#     output.set_shape(shape)
#     return output
#
#
# def _swap_axis(logits, dim_index, last_index, **kwargs):
#     return tf.transpose(
#         logits,
#         tf.concat(
#             [
#                 tf.range(dim_index),
#                 [last_index],
#                 tf.range(dim_index + 1, last_index),
#                 [dim_index],
#             ],
#             0,
#         ),
#         **kwargs,
#     )
#
#
# def _compute_2d_sparsemax(logits):
#     """Performs the sparsemax operation when axis=-1."""
#     shape_op = tf.shape(logits)
#     obs = tf.math.reduce_prod(shape_op[:-1])
#     dims = shape_op[-1]
#
#     # In the paper, they call the logits z.
#     # The mean(logits) can be substracted from logits to make the algorithm
#     # more numerically stable. the instability in this algorithm comes mostly
#     # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
#     # to zero. However, in practise the numerical instability issues are very
#     # minor and substacting the mean causes extra issues with inf and nan
#     # input.
#     # Reshape to [obs, dims] as it is almost free and means the remanining
#     # code doesn't need to worry about the rank.
#     z = tf.reshape(logits, [obs, dims])
#
#     # sort z
#     z_sorted, _ = tf.nn.top_k(z, k=dims)
#
#     # calculate k(z)
#     z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
#     k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
#     z_check = 1 + k * z_sorted > z_cumsum
#     # because the z_check vector is always [1,1,...1,0,0,...0] finding the
#     # (index + 1) of the last `1` is the same as just summing the number of 1.
#     k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)
#
#     # calculate tau(z)
#     # If there are inf values or all values are -inf, the k_z will be zero,
#     # this is mathematically invalid and will also cause the gather_nd to fail.
#     # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
#     # fixed later (see p_safe) by returning p = nan. This results in the same
#     # behavior as softmax.
#     k_z_safe = tf.math.maximum(k_z, 1)
#     indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1],
#                        axis=1)
#     tau_sum = tf.gather_nd(z_cumsum, indices)
#     tau_z = (tau_sum - 1) / tf.cast(k_z_safe, logits.dtype)
#
#     # calculate p
#     p = tf.math.maximum(tf.cast(0, logits.dtype),
#                         z - tf.expand_dims(tau_z, -1))
#     # If k_z = 0 or if z = nan, then the input is invalid
#     p_safe = tf.where(
#         tf.expand_dims(
#             tf.math.logical_or(tf.math.equal(k_z, 0),
#                                tf.math.is_nan(z_cumsum[:, -1])),
#             axis=-1,
#         ),
#         # tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
#         tf.math.softmax(z),  # fallback to softmax instead of returning nan
#         p,
#     )
#
#     # Reshape back to original size
#     p_safe = tf.reshape(p_safe, shape_op)
#     return p_safe
#############################
