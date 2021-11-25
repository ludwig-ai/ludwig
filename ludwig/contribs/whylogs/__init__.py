from ludwig.callbacks import Callback
from whylogs import get_or_create_session


class WhyLogsCallback(Callback):
    def __init__(self, path_to_config=None):
        self.session = get_or_create_session(path_to_config)

    def on_build_metadata_start(self, df, mode=None):
        def log_dataframe(df_aux):
            session = get_or_create_session()
            session.log_dataframe(
                df_aux,
                mode,
                tags={"stage": "build_metadata_start", "mode": mode if not mode else ""}
            )

        if hasattr(df, "compute"):
            df.map_partitions(log_dataframe).compute()
        else:
            log_dataframe(df)
