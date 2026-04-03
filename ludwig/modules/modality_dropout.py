"""Modality dropout: randomly drop encoder outputs during training for robustness.

During training, each input feature's encoder output is randomly replaced with a learnable
"missing modality" embedding with probability `dropout_prob`. This improves robustness to
missing inputs at inference time.

Based on: "Bag of Tricks for Multimodal AutoML" (arXiv 2412.16243, Dec 2024).
"""

import torch
import torch.nn as nn

from ludwig.constants import ENCODER_OUTPUT
from ludwig.utils.torch_utils import LudwigModule


class ModalityDropout(LudwigModule):
    """Drops entire modality encoder outputs during training and replaces with learned embeddings."""

    def __init__(self, feature_shapes: dict[str, torch.Size], dropout_prob: float = 0.1):
        """Initialize modality dropout.

        Args:
            feature_shapes: Dict mapping feature names to their encoder output shapes.
            dropout_prob: Probability of dropping each feature's output during training.
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.missing_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(*shape)) for name, shape in feature_shapes.items()}
        )

    def forward(self, encoder_outputs: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        if not self.training or self.dropout_prob == 0.0:
            return encoder_outputs

        result = {}
        for name, output_dict in encoder_outputs.items():
            if name in self.missing_embeddings and torch.rand(1).item() < self.dropout_prob:
                batch_size = output_dict[ENCODER_OUTPUT].shape[0]
                missing = (
                    self.missing_embeddings[name].unsqueeze(0).expand(batch_size, *self.missing_embeddings[name].shape)
                )
                result[name] = {**output_dict, ENCODER_OUTPUT: missing}
            else:
                result[name] = output_dict
        return result

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([len(self.missing_embeddings)])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape
