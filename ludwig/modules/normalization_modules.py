import torch

from ludwig.utils.torch_utils import LudwigModule


class GhostBatchNormalization(LudwigModule):
    def __init__(
            self,
            num_features: int,
            momentum: float = 0.9,
            epsilon: float = 1e-3,
            virtual_batch_size: int = None
    ):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = torch.nn.BatchNorm1d(num_features, momentum=momentum,
                                       eps=epsilon)

    def forward(self, inputs):
        if self.training and self.virtual_batch_size:
            batch_size = inputs.shape[0]

            q, r = divmod(batch_size, self.virtual_batch_size)
            num_or_size_splits = q
            if r != 0:
                num_or_size_splits = [self.virtual_batch_size] * q + [r]

            splits = torch.split(inputs, num_or_size_splits)
            x = [self.bn(inputs) for x in splits]
            return torch.cat(x, 0)

        return self.bn(inputs)

    @property
    def moving_mean(self):
        return self.bn.running_mean

    @property
    def moving_variance(self):
        return self.bn.running_var
