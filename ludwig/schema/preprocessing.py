from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import RANDOM
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import PREPROCESSING_METADATA
from ludwig.schema.split import BaseSplitConfig, SplitDataclassField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class PreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Global preprocessing config is a dataclass that configures the parameters used for global preprocessing."""

    sample_ratio: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="The ratio of the dataset to use. For instance, if 0.5, half of the dataset "
        "provided will be used.",
        parameter_metadata=PREPROCESSING_METADATA["sample_ratio"],
    )

    sample_size: float = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="The maximum number of samples from the dataset to use. Cannot be set if sample_ratio is set to be "
        "< 1.0. If sample_ratio is set to 1.0, this will override the number of samples to used.",
        parameter_metadata=PREPROCESSING_METADATA["sample_size"],
    )

    eval_steps: float = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="The number of steps to use for evaluation. If None, the entire dataset will be used.",
        parameter_metadata=PREPROCESSING_METADATA["eval_steps"],
    )

    oversample_minority: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="If not None, the minority class will be oversampled to reach the specified ratio respective to "
        "the majority class. ",
        parameter_metadata=PREPROCESSING_METADATA["oversample_minority"],
    )

    undersample_majority: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="If not None, the majority class will be undersampled to reach the specified ratio respective "
        "to the minority class. ",
        parameter_metadata=PREPROCESSING_METADATA["undersample_majority"],
    )

    split: BaseSplitConfig = SplitDataclassField(
        default=RANDOM,
    )

    global_max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Specifically for LLMs. This is the maximum length of the input sequence going into the model's "
        "forward pass during training. Sequences will be truncated to this length after merging inputs and targets. "
        "If not set, the total length of the merged input and target token sequences will be used.",
        parameter_metadata=PREPROCESSING_METADATA["global_max_sequence_length"],
    )


@DeveloperAPI
class PreprocessingField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(PreprocessingConfig)
