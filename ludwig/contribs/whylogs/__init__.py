from ludwig.callbacks import Callback
from whylogs import get_or_create_session


class WhyLogsCallback(Callback):
    def __init__(self, path_to_config=None):
        self.session = get_or_create_session(path_to_config)

    def on_build_metadata_start(self, df, dataset_name=None):
        self.session.log_dataframe(df, dataset_name, tags={"stage": "build_metadata_start"})

    def on_build_metadata_end(self, df, dataset_name=None):
        self.session.log_dataframe(df, dataset_name, tags={"stage": "build_metadata_end"})

    def on_build_data_start(self, df, dataset_name=None):
        self.session.log_dataframe(df, dataset_name, tags={"stage": "build_data_end"})

    def on_build_data_end(self, df, dataset_name=None):
        self.session.log_dataframe(df, dataset_name, tags={"stage": "build_data_end"})
