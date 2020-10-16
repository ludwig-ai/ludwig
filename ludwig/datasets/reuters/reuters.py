import pandas as pd
from ludwig.datasets.base_dataset import BaseDataset

"""The Reuters dataset which for now mostly relies on its parent to perform the
major parts of the workflow which include download/process/load"""


class Reuters(BaseDataset):

    def __init__(self, cache_location):
        super().__init__(dataset_name="reuters", cache_location=cache_location)

    def download(self) -> None:
        super().download()

    def process(self) -> None:
        super().process()

    def load(self) -> pd.DataFrame:
        return super().load()