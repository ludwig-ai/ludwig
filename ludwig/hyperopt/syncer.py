from typing import Callable, Dict, List, Optional, Tuple

from ray.tune.syncer import _BackgroundSyncer

from ludwig.backend import Backend
from ludwig.utils.data_utils import use_credentials
from ludwig.utils.fs_utils import delete, download, upload


class RemoteSyncer(_BackgroundSyncer):
    def __init__(self, sync_period: float = 300.0, backend: Backend = None):
        super().__init__(sync_period=sync_period)
        self.backend = backend

    def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List] = None) -> Tuple[Callable, Dict]:
        with use_credentials(self.backend.hyperopt_sync_manager.credentials):
            return upload(), dict(lpath=local_path, rpath=uri, recursive=True)

    def _sync_down_command(self, uri: str, local_path: str) -> Tuple[Callable, Dict]:
        with use_credentials(self.backend.hyperopt_sync_manager.credentials):
            return download, dict(rpath=uri, lpath=local_path, recursive=True)

    def _delete_command(self, uri: str) -> Tuple[Callable, Dict]:
        with use_credentials(self.backend.hyperopt_sync_manager.credentials):
            return delete(), dict(url=uri, recursive=True)

    def __reduce__(self):
        """We need this custom serialization because we can't pickle thread.lock objects that are used by the
        use_credentials context manager.

        https://docs.ray.io/en/latest/ray-core/objects/serialization.html#customized-serialization
        """
        deserializer = RemoteSyncer
        serialized_data = (self.sync_period, self.backend)
        return deserializer, serialized_data
