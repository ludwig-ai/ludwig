from typing import Any, Dict, Optional

from ray import tune
from ray.tune.syncer import get_node_to_storage_syncer, Syncer

from ludwig.utils.data_utils import use_credentials
from ludwig.utils.misc_utils import memoized_method


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
