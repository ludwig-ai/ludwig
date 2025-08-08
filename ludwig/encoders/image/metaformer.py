#! /usr/bin/env python
# Copyright (c) 2025
#
# New MetaFormer / CAFormer style image encoder for Ludwig.
#
# This integrates ConvFormer / CAFormer family backbones as a first-class
# Ludwig encoder, avoiding runtime monkey patching of existing encoders.
#
# The implementation wraps the existing CAFormerStackedCNN (renamed conceptually
# to MetaFormerStackedCNN) logic currently living in caformer_setup_backup.
#
# TODO (follow-up in PR):
# - Move / refactor caformer_setup_backup/ code into ludwig/modules or a
#   dedicated ludwig/encoders/image/metaformer_backbones.py file.
# - Add a proper schema definition (see get_schema_cls TODO below).
# - Add unit tests under tests/ludwig/encoders/test_metaformer_encoder.py
# - Add release note and documentation snippet.

from typing import Any, Dict, Optional, Type
import logging

import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("metaformer", IMAGE)
class MetaFormerEncoder(Encoder):
    """MetaFormerEncoder
    Provides access to MetaFormer / CAFormer style convolution-attention hybrids
    (e.g., caformer_s18 / m36 / b36 etc.) as a Ludwig image encoder.

    Configuration (proposed):
        type: metaformer
        model_name: caformer_s18        # required
        use_pretrained: true            # optional (default True)
        trainable: true                 # optional
        output_size: 128                # dimensionality after projection head

    Behavior:
        - Loads the specified backbone.
        - Adapts input spatial size & channels as needed (handled by backbone wrapper).
        - Emits a dense representation of shape (output_size,).
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int = 3,
        model_name: Optional[str] = None,
        use_pretrained: bool = True,
        trainable: bool = True,
        output_size: int = 128,
        encoder_config=None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = encoder_config
        self.model_name = model_name or "caformer_s18"
        self.use_pretrained = use_pretrained
        self.trainable = trainable
        self.output_size = output_size

        # Import existing implementation (currently in backup namespace).
        # In a polished PR this code should be relocated inside core tree.
        try:
            # Updated to use integrated metaformer implementation
            from metaformer_integration.metaformer_stacked_cnn import MetaFormerStackedCNN as _BackboneWrapper
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import CAFormer / MetaFormer backbone code. "
                "Ensure integration code is migrated from caformer_setup_backup."
            ) from e

        logger.info(
            "Initializing MetaFormerEncoder backbone=%s pretrained=%s trainable=%s output_size=%d",
            self.model_name,
            self.use_pretrained,
            self.trainable,
            self.output_size,
        )

        self.backbone_wrapper: nn.Module = _BackboneWrapper(
            height=height if height is not None else 224,
            width=width if width is not None else 224,
            num_channels=num_channels if num_channels is not None else 3,
            output_size=output_size,
            custom_model=self.model_name,
            use_pretrained=self.use_pretrained,
            trainable=self.trainable,
        )

        # Expose shapes
        self._input_shape = (num_channels, height, width)
        # Backbone wrapper exposes output_shape as list -> convert to torch.Size
        raw_out_shape = getattr(self.backbone_wrapper, "output_shape", [self.output_size])
        if isinstance(raw_out_shape, (list, tuple)):
            self._output_shape = torch.Size(raw_out_shape)
        else:
            self._output_shape = torch.Size([self.output_size])

        # Freeze if not trainable
        if not self.trainable:
            for p in self.backbone_wrapper.parameters():
                p.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        # Expect shape: [B, C, H, W]
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("MetaFormerEncoder forward expects a torch.Tensor input.")
        out_dict = self.backbone_wrapper(inputs)
        if isinstance(out_dict, dict) and "encoder_output" in out_dict:
            rep = out_dict["encoder_output"]
        else:
            # Fallback: treat raw module output as representation
            rep = out_dict
        return {ENCODER_OUTPUT: rep}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape

    @staticmethod
    def get_schema_cls() -> Type[Any]:
        # Return dedicated MetaFormerConfig (added in schema tree)
        try:
            from ludwig.schema.encoders.image.metaformer import MetaFormerConfig
            return MetaFormerConfig  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("MetaFormerConfig schema import failed.") from e
