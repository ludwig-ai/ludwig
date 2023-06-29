from typing import Required, TypedDict

import torch


class EncoderOutputDict(TypedDict, total=False):
    encoder_output: Required[torch.Tensor]
    encoder_output_state: torch.Tensor
    attentions: torch.Tensor
