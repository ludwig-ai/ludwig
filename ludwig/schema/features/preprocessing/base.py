from typing import ClassVar, Optional

from ludwig.schema import utils as schema_utils


class BasePreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for input feature preprocessing. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding input feature class are copied over: check each class to check which attributes are different
    from the preprocessing of each feature.
    """

    feature_type: ClassVar[Optional[str]] = None
    "Class variable pointing to the corresponding preprocessor."
