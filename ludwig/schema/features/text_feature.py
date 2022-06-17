from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class TextInputFeatureConfig(base.BaseFeatureConfig):
    """
    TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='text'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "embed", "parallel_cnn", "stacked_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn", "transformer",
         "albert", "mt5", "xlmroberta", "bert", "xlm", "gpt", "gpt2", "roberta", "transformer_xl", "xlnet",
         "distilbert", "ctrl", "camembert", "t5", "flaubert", "electra", "longformer", "auto_transformer"],
        default="parallel_cnn",
        description="Encoder to use for this text feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class TextOutputFeatureConfig(base.BaseFeatureConfig):
    """
    TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        ["tagger", "generator"],
        default="generator",
        description="Decoder to use for this text output feature.",
    )
