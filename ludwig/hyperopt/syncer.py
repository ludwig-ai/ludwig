from typing import Any, Callable, Dict, List, Optional, Tuple

from ray import tune
from ray.tune.syncer import _BackgroundSyncer, get_node_to_storage_syncer, Syncer

from ludwig.utils.data_utils import use_credentials
from ludwig.utils.fs_utils import delete, download, upload
from ludwig.utils.misc_utils import memoized_method


class RemoteSyncer(_BackgroundSyncer):
    def __init__(self, sync_period: float = 300.0, creds: Optional[Dict[str, Any]] = None):
        super().__init__(sync_period=sync_period)
        self.creds = creds

    def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List] = None) -> Tuple[Callable, Dict]:
        with use_credentials(self.creds):
            return upload, dict(lpath=local_path, rpath=uri)

    def _sync_down_command(self, uri: str, local_path: str) -> Tuple[Callable, Dict]:
        with use_credentials(self.creds):
            return download, dict(rpath=uri, lpath=local_path)

    def _delete_command(self, uri: str) -> Tuple[Callable, Dict]:
        with use_credentials(self.creds):
            return delete, dict(url=uri, recursive=True)

    def __reduce__(self):
        """We need this custom serialization because we can't pickle thread.lock objects that are used by the
        use_credentials context manager.

        https://docs.ray.io/en/latest/ray-core/objects/serialization.html#customized-serialization
        """
        deserializer = RemoteSyncer
        serialized_data = (self.sync_period, self.creds)
        return deserializer, serialized_data


class LazyFsspecSyncer(Syncer):
    def __init__(self, upload_dir: str, creds: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.upload_dir = upload_dir
        self.creds = creds
        self._syncer = None

    def sync_up(self, *args, **kwargs) -> bool:
        with use_credentials(self.creds):
            return self.syncer().sync_up(*args, **kwargs)

    def sync_down(self, *args, **kwargs) -> bool:
        with use_credentials(self.creds):
            return self.syncer().sync_down(*args, **kwargs)

    def delete(self, *args, **kwargs) -> bool:
        with use_credentials(self.creds):
            return self.syncer().delete(*args, **kwargs)

    @memoized_method(maxsize=1)
    def syncer(self):
        if self._syncer is None:
            sync_config = tune.SyncConfig(upload_dir=self.upload_dir)
            self._syncer = get_node_to_storage_syncer(sync_config)
        return self._syncer


class WrappedSyncer(Syncer):
    def __init__(self, syncer: Syncer, creds: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.syncer = syncer
        self.creds = creds

    def sync_up(self, *args, **kwargs) -> bool:
        with use_credentials(self.creds):
            return self.syncer.sync_up(*args, **kwargs)

    def sync_down(self, *args, **kwargs) -> bool:
        with use_credentials(self.creds):
            return self.syncer.sync_down(*args, **kwargs)

    def delete(self, *args, **kwargs) -> bool:
        with use_credentials(self.creds):
            return self.syncer.delete(*args, **kwargs)
