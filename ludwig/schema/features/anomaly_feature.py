# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ANOMALY, ANOMALY_AUROC, DEEP_SVDD, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.features.base import BaseOutputFeatureConfig
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.utils import LossDataclassField
from ludwig.schema.features.utils import (
    ecd_defaults_config_registry,
    ecd_output_config_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import LudwigBaseConfig


@DeveloperAPI
@output_mixin_registry.register(ANOMALY)
class AnomalyOutputFeatureConfigMixin(LudwigBaseConfig):
    """AnomalyOutputFeatureConfigMixin configures parameters shared between the anomaly output feature and the
    anomaly global defaults section of the Ludwig config."""

    decoder: BaseDecoderConfig = None

    loss: BaseLossConfig = LossDataclassField(
        feature_type=ANOMALY,
        default=DEEP_SVDD,
    )


@DeveloperAPI
class AnomalyOutputFeatureConfig(AnomalyOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """AnomalyOutputFeatureConfig configures the parameters for the anomaly output feature.

    The anomaly output feature implements Deep One-Class Classification: the encoder maps
    all inputs into a latent space, and the decoder computes the squared Euclidean distance
    from a learned hypersphere center c. This distance is the *anomaly score* — the higher
    the score, the more anomalous the input.

    Three loss functions are available:

    - ``deep_svdd`` (default): Geometric hypersphere objective. Pulls all training points
      toward center c. Simple, interpretable, and effective for homogeneous normal data.
      Ruff et al., ICML 2018.

    - ``deep_sad``: Semi-supervised extension. Requires a target column with 0 (normal),
      1 (confirmed anomaly), or -1 (unlabeled). Labeled anomalies are pushed *away* from c
      while normal/unlabeled samples are pulled toward it.
      Ruff et al., ICLR 2020.

    - ``drocc``: Adversarially robust variant. Adds a perturbation-based regularizer to
      prevent hypersphere collapse — a degenerate solution where all representations
      converge to c. Recommended when using expressive encoders (e.g. transformer-based).
      Goyal et al., ICML 2020.

    **Multimodal anomaly detection** works out of the box: simply add multiple input features
    (text, image, tabular, audio, etc.) to the ECD model. The combiner will fuse them before
    the anomaly decoder.

    **Threshold selection**: after training, the ``threshold`` determines when an anomaly
    score is classified as an anomaly. Set ``threshold="auto"`` to automatically select
    the threshold as the `threshold_percentile`-th percentile of validation scores
    (e.g., 95th percentile means 5% of validation examples are flagged).
    """

    type: str = schema_utils.ProtectedString(ANOMALY)

    default_validation_metric: str = schema_utils.StringOptions(
        [ANOMALY_AUROC],
        default=ANOMALY_AUROC,
        description="Internal only: default validation metric for anomaly output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description=(
            "How to reduce an input that is not a vector, but a matrix or higher-order tensor, on the first dimension "
            "(second if you count the batch dimension)."
        ),
    )

    threshold: float | str = schema_utils.OneOfOptionsField(
        default="auto",
        description=(
            "Decision threshold for classifying a sample as anomalous. Anomaly scores above this value are "
            "predicted as anomalies. Set to ``'auto'`` to automatically select the threshold as the "
            "``threshold_percentile``-th percentile of anomaly scores on the validation set. "
            "Set to a float in [0, inf) to use a fixed threshold."
        ),
        field_options=[
            schema_utils.FloatRange(default=0.5, min=0.0),
            schema_utils.StringOptions(["auto"], default="auto"),
        ],
    )

    threshold_percentile: float = schema_utils.FloatRange(
        default=95.0,
        min=0.0,
        max=100.0,
        description=(
            "When ``threshold='auto'``, this is the percentile of validation-set anomaly scores used as the "
            "decision threshold. For example, 95.0 means 5% of validation examples are flagged as anomalies. "
            "Ignored when ``threshold`` is a fixed float."
        ),
    )


@DeveloperAPI
@ecd_output_config_registry.register(ANOMALY)
class ECDAnomalyOutputFeatureConfig(AnomalyOutputFeatureConfig):
    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=ANOMALY,
        default="anomaly",
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(ANOMALY)
class AnomalyDefaultsConfig(AnomalyOutputFeatureConfigMixin):
    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=ANOMALY,
        default="anomaly",
    )
