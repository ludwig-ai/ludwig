import logging
from dataclasses import dataclass

from dataclasses_json import LetterCase, dataclass_json
from pandas import Series

try:
    import ray
except ImportError:
    raise ImportError(
        ' ray is not installed. '
        'In order to use auto_train please run '
        'pip install ludwig[ray]'
    )


logger = logging.getLogger(__name__)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldInfo:
    name: str
    dtype: str
    key: str = None
    distinct_values: int = 0
    nonnull_values: int = 0
    image_values: int = 0
    avg_words: int = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldConfig:
    name: str
    column: str
    type: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldMetadata:
    name: str
    config: FieldConfig
    excluded: bool
    mode: str
    missing_values: float


def avg_num_tokens(field: Series) -> int:
    # sample a subset if dataframe is large
    if len(field) > 5000:
        field = field.sample(n=5000, random_state=40)
    unique_entries = field.unique()
    avg_words = Series(unique_entries).str.split().str.len().mean()
    return avg_words


def get_available_resources():
    # returns total number of gpus and cpus
    resources = ray.cluster_resources()
    gpus = resources.get('GPU', 0)
    cpus = resources.get('CPU', 0)
    resources = {
        'gpu': gpus,
        'cpu': cpus
    }
    return resources


def _ray_init():
    if ray.is_initialized():
        return

    try:
        ray.init('auto', ignore_reinit_error=True)
    except ConnectionError:
        logger.info('Initializing new Ray cluster...')
        ray.init()
