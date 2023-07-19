from typing import TypedDict

import torch


class EncoderOutputDict(TypedDict, total=False):
    encoder_output: torch.Tensor
    encoder_output_state: torch.Tensor  # only used by sequence and h3 encoders
    attentions: torch.Tensor  # only used by the vit legacy encoder
