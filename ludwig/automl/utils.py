from dataclasses import dataclass

import GPUtil
import psutil
from dataclasses_json import LetterCase, dataclass_json
from pandas import Series


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldInfo:
    name: str
    dtype: str
    key: str = None
    distinct_values: int = 0
    nonnull_values: int = 0
    avg_words: int = None


def avg_num_tokens(field: Series) -> int:
    # sample a subset if dataframe is large
    if len(field) > 5000:
        field = field.sample(n=5000, random_state=40)
    unique_entries = field.unique()
    avg_words = Series(unique_entries).str.split().str.len().mean()
    return avg_words


def get_available_resources():
    # returns total number of gpus and cpus
    GPUs = GPUtil.getGPUs()
    GPUavailability = GPUtil.getAvailability(
        GPUs, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    CPUs = psutil.cpu_count()
    resources = {
        'gpu': sum(GPUavailability),
        'cpu': CPUs
    }
    return resources
