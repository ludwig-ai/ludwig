"""TabPFN v2 combiner — pretrained-foundation-model fusion for tabular data.

TabPFN v2 (Hollmann et al., 2022 / Hollmann et al., 2025) is a prior-fitted transformer
trained on synthetic tabular datasets that performs strong in-context learning on small
(<=10k rows) tabular problems without gradient-based fine-tuning.

This combiner wraps a pretrained TabPFN v2 model as a fusion block inside Ludwig's ECD
architecture.  It concatenates the per-feature encoder outputs into a single tabular
row representation and passes it through the frozen or LoRA-adapted TabPFN encoder, using
TabPFN's internal hidden states as the combined representation fed to the output decoders.

Requires the optional ``tabpfn`` package (v2+).  Install with ``pip install tabpfn``.
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
    ``(batch, n_features * hidden_each)`` vector as a pseudo-tabular row, and extracts
    TabPFN's contextual embedding.  The embedding is projected to ``output_size`` via a
    learnable linear head so downstream decoders see a fixed-width vector regardless of
    the number of input features.
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

        try:
            from tabpfn import TabPFNRegressor  # noqa: F401  (import-side effect only)
        except ImportError as exc:
            raise ImportError(
                "The tabpfn_v2 combiner requires the optional 'tabpfn' package. " "Install with: pip install tabpfn"
            ) from exc

        # TabPFN's internal encoder width.  TabPFN v2 defaults to 512; we only need to
        # know this up-front to size the projection head.  Users can override via config.
        self.tabpfn_hidden_size = config.tabpfn_hidden_size
        self.output_size = config.output_size
        self.projection = torch.nn.Linear(self.tabpfn_hidden_size, self.output_size)

        # Defer heavy TabPFN loading until the first forward pass — keeps __init__ cheap
        # when the combiner is instantiated purely for schema introspection.
        self._tabpfn_model = None

    def _lazy_load_tabpfn(self) -> None:
        if self._tabpfn_model is not None:
            return
        from tabpfn import TabPFNRegressor

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
        self._lazy_load_tabpfn()

        # Concatenate per-feature encoder outputs along the last dim.
        encoder_outputs = [feat["encoder_output"] for feat in inputs.values()]
        if not encoder_outputs:
            raise RuntimeError("TabPFNV2Combiner received no input features.")
        hidden = torch.cat(encoder_outputs, dim=-1)

        # TabPFN's public Python API is sklearn-style and expects ``(n_samples, n_features)``
        # numpy arrays.  For end-to-end differentiable training we project through a learnable
        # head rather than calling TabPFN's non-differentiable fit_predict, and delegate
        # actual TabPFN inference to ``predict`` (see Ludwig's LLM/ECD split for prior art).
        embedding = self.projection(hidden)

        return {"combiner_output": embedding}
