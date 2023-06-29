from typing import TypedDict

import torch


class EncoderOutputDict(TypedDict, total=False):
    encoder_output: torch.Tensor
    encoder_output_state: torch.Tensor
    attentions: torch.Tensor
