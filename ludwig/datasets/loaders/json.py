import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class JsonLoader(DatasetLoader):
    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path)
