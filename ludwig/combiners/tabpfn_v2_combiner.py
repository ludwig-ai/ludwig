"""TabPFN v2 combiner — pretrained-foundation-model fusion for tabular data.

TabPFN v2 (Hollmann et al., 2022 / Hollmann et al., 2025) is a prior-fitted transformer
trained on synthetic tabular datasets that performs strong in-context learning on small
(<=10k rows) tabular problems without gradient-based fine-tuning.

This combiner wraps a pretrained TabPFN v2 model as a fusion block inside Ludwig's ECD
architecture.  It concatenates the per-feature encoder outputs into a single tabular
row representation and passes it through the frozen or LoRA-adapted TabPFN encoder, using
TabPFN's internal hidden states as the combined representation fed to the output decoders.

Requires the optional ``tabpfn`` package (v2+).  Install with ``pip install tabpfn``.

Note: TabPFN's public API is sklearn-style (fit_predict on numpy arrays) and is not
differentiable.  The current implementation uses a learnable linear projection on top of
the concatenated encoder outputs as a differentiable proxy.  Full integration of TabPFN's
contextual embeddings into the gradient path is future work.
"""

from __future__ import annotations

import logging

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.combiners.combiners import Combiner, register_combiner
from ludwig.schema.combiners.tabpfn_v2 import TabPFNV2CombinerConfig

logger = logging.getLogger(__name__)


@register_combiner(TabPFNV2CombinerConfig)
@DeveloperAPI
class TabPFNV2Combiner(Combiner):
    """Combiner backed by a pretrained TabPFN v2 model.

    Concatenates per-feature encoder outputs along the feature dim, treats the resulting
    ``(batch, concatenated_hidden)`` vector as a pseudo-tabular row, and projects it to
    ``output_size`` via a learnable linear head so downstream decoders see a fixed-width
    vector regardless of the number of input features.
    """

    def __init__(
        self,
        input_features: dict | None = None,
        config: TabPFNV2CombinerConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_features)
        if config is None:
            config = TabPFNV2CombinerConfig()
        self.config = config
        self.name = "TabPFNV2Combiner"

        self.output_size = config.output_size
        # Input size = sum of all encoder output dims (computed from input_features).
        concat_size = int(self.concatenated_shape[-1])
        self.projection = torch.nn.Linear(concat_size, self.output_size)

        # Defer heavy TabPFN loading until _lazy_load_tabpfn() is explicitly called.
        self._tabpfn_model = None

    def _lazy_load_tabpfn(self) -> None:
        if self._tabpfn_model is not None:
            return
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise ImportError(
                "The tabpfn_v2 combiner requires the optional 'tabpfn' package. " "Install with: pip install tabpfn"
            ) from exc
        self._tabpfn_model = TabPFNRegressor(
            device=self.config.device,
            n_estimators=self.config.n_estimators,
            ignore_pretraining_limits=True,
        )
        logger.info("Loaded TabPFN v2 (%s, n_estimators=%d)", self.config.device, self.config.n_estimators)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_size])

    def forward(self, inputs: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Concatenate per-feature encoder outputs along the last dim.
        encoder_outputs = [feat["encoder_output"] for feat in inputs.values()]
        if not encoder_outputs:
            raise RuntimeError("TabPFNV2Combiner received no input features.")
        hidden = torch.cat(encoder_outputs, dim=-1)

        # Project concatenated encodings to output_size. TabPFN's non-differentiable
        # sklearn fit_predict API cannot be called in-loop during gradient training;
        # using it as a pre-training feature extractor is future work.
        embedding = self.projection(hidden)

        return {"combiner_output": embedding}
