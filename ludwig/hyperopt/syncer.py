from typing import Any, Callable, Dict, List, Optional, Tuple

from ray.tune.syncer import _BackgroundSyncer

from ludwig.utils.data_utils import use_credentials
from ludwig.utils.fs_utils import delete, download, upload


class RemoteSyncer(_BackgroundSyncer):
    def __init__(self, sync_period: float = 300.0, creds: Optional[Dict[str, Any]] = None):
        super().__init__(sync_period=sync_period)
        self.creds = creds

    def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List] = None) -> Tuple[Callable, Dict]:
        def upload_cmd(*args, **kwargs):
            with use_credentials(self.creds):
                return upload(*args, **kwargs)

        return upload_cmd, dict(lpath=local_path, rpath=uri)

    def _sync_down_command(self, uri: str, local_path: str) -> Tuple[Callable, Dict]:
        def download_cmd(*args, **kwargs):
            with use_credentials(self.creds):
                return download(*args, **kwargs)

        return download_cmd, dict(rpath=uri, lpath=local_path)

    def _delete_command(self, uri: str) -> Tuple[Callable, Dict]:
        def delete_cmd(*args, **kwargs):
            with use_credentials(self.creds):
                return delete(*args, **kwargs)

        return delete_cmd, dict(url=uri, recursive=True)

    def __reduce__(self):
        """We need this custom serialization because we can't pickle thread.lock objects that are used by the
        use_credentials context manager.

        https://docs.ray.io/en/latest/ray-core/objects/serialization.html#customized-serialization
        """
        deserializer = RemoteSyncer
        serialized_data = (self.sync_period, self.creds)
        return deserializer, serialized_data
