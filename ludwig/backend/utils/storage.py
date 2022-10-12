import contextlib
from typing import Any, Dict, Optional, Union

from ludwig.utils import data_utils
from ludwig.utils.fs_utils import RemoteFilesystem

CredInputs = Optional[Union[str, Dict[str, Any]]]


DEFAULTS = "defaults"
ARTIFACTS = "artifacts"
DATASETS = "datasets"
CACHE = "cache"


class Storage:
    def __init__(self, creds: Optional[Dict[str, Any]]):
        self._creds = creds

    @property
    def fs(self) -> RemoteFilesystem:
        return RemoteFilesystem(self._creds)

    @contextlib.contextmanager
    def use(self):
        with data_utils.use_credentials(self._creds):
            yield

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return self._creds


class StorageManager:
    def __init__(
        self,
        defaults: CredInputs = None,
        artifacts: CredInputs = None,
        datasets: CredInputs = None,
        cache: CredInputs = None,
    ):
        defaults = load_creds(defaults)
        cred_inputs = {
            DEFAULTS: defaults,
            ARTIFACTS: load_creds(artifacts),
            DATASETS: load_creds(datasets),
            CACHE: load_creds(cache),
        }

        self.storages = {k: Storage(v if v is not None else defaults) for k, v in cred_inputs.items()}

    @property
    def defaults(self) -> Storage:
        return self.storages[DEFAULTS]

    @property
    def artifacts(self) -> Storage:
        return self.storages[ARTIFACTS]

    @property
    def datasets(self) -> Storage:
        return self.storages[DATASETS]

    @property
    def cache(self) -> Storage:
        return self.storages[CACHE]


def load_creds(cred: CredInputs) -> Dict[str, Any]:
    if isinstance(cred, str):
        cred = data_utils.load_json(cred)
    return cred
