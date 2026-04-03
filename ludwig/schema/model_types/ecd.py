from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import CombinerSelection
from ludwig.schema.defaults.ecd import ECDDefaultsConfig, ECDDefaultsField
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    ECDInputFeatureSelection,
    ECDOutputFeatureSelection,
    FeatureCollection,
)
from ludwig.schema.hyperopt import HyperoptConfig, HyperoptField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import ECDTrainerConfig, ECDTrainerField


@DeveloperAPI
@register_model_type(name="ecd")
class ECDModelConfig(ModelConfig):
    """Parameters for ECD."""

    model_type: str = schema_utils.ProtectedString("ecd")

    preset: str | None = schema_utils.StringOptions(
        options=["medium_quality", "high_quality", "best_quality"],
        default=None,
        allow_none=True,
        description=(
            "Quality preset that sets sensible defaults for combiner, trainer, and other settings. "
            "User-specified values always take precedence. "
            "'medium_quality': fast training with concat combiner. "
            "'high_quality': transformer combiner with uncertainty loss balancing. "
            "'best_quality': FT-Transformer, uncertainty loss balancing, model soup."
        ),
    )

    input_features: FeatureCollection[BaseInputFeatureConfig] = ECDInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = ECDOutputFeatureSelection().get_list_field()

    combiner: BaseCombinerConfig = CombinerSelection().get_default_field()

    trainer: ECDTrainerConfig = ECDTrainerField().get_default_field()
    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: ECDDefaultsConfig = ECDDefaultsField().get_default_field()
    hyperopt: HyperoptConfig | None = HyperoptField().get_default_field()
