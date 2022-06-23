from ludwig.callbacks import Callback
from ludwig.utils.package_utils import LazyLoader

whylogs = LazyLoader("whylogs", globals(), "whylogs")


class WhyLogsCallback(Callback):
    def __init__(self, path_to_config=None):
        self.path_to_config = path_to_config

    def on_build_metadata_start(self, df, mode=None):
        def log_dataframe(df_aux):
            session = whylogs.get_or_create_session(self.path_to_config)
            session.log_dataframe(df_aux, mode, tags={"stage": "build_metadata_start", "mode": mode if mode else ""})

        if hasattr(df, "compute"):
            df.map_partitions(log_dataframe).compute()
        else:
            log_dataframe(df)
