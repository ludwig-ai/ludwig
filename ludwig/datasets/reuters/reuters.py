import csv
import os
import yaml
import shutil
from pathlib import Path
import pandas as pd
from ludwig.datasets.base_dataset import BaseDataset

"""A class to download, process and return the ohsumed training data to be plugged into the Ludwig training API"""


class Reuters(BaseDataset):

    def __init__(self):
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path, "../text/versions.yaml")
        self._source_location = os.path.join(self._initial_path, "../text/reuters/reuters-allcats.csv")
        self._cache_dir = str(Path.home().joinpath('.ludwig_cache'))
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self._cur_version = self._config_file_contents["text"]["reuters"]
        self._artifact_dir = str(Path.home().joinpath('.ludwig_cache').joinpath("reuters_"+str(self._cur_version)))
        if not os.path.exists(self._artifact_dir):
            os.makedirs(self._artifact_dir)
        self._result_dict={}

    """Download the raw data to the ludwig cache in the format ~/.ludwig_cache/id/raw.csv
    where id is represented by the name.version of the dataset
    args:
        dataset_name: The name of the dataset we need to retrieve
    return: none
    """
    def download(self, dataset_name) -> None:
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("reuters_"
                                                       + str(self._cur_version)).joinpath('raw.csv')
        shutil.copy(self._source_location, download_path)
        result = os.path.isfile(download_path)
        if not result:
            raise FileNotFoundError("The raw reuters data was not downloaded correctly")

    """Transform the raw dataset into a format that can be used by the ludwig
       training API
       args:
           None
       :returns:
           None
    """
    def process(self) -> None:
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("reuters_"
                                                                       + str(self._cur_version)).joinpath('raw.csv')
        processed_data_path = Path.home().joinpath('.ludwig_cache')\
            .joinpath("reuters_"+ str(self._cur_version)).joinpath('processed.csv')
        result = os.path.isfile(download_path)
        if result:
            dict_reader = csv.DictReader(open(download_path))
            value_to_store = None
            for row in dict_reader:
                for key, value in row.items():
                    if key == "class":
                        value_to_store = value
                    else:
                        key_to_store = value
                        self._result_dict[key_to_store] = value_to_store
        try:
            with open(processed_data_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self._result_dict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

    """Now that the data is processed load and return it as a pandas dataframe
     Args:
         self (BaseDataset): A handle to the current class
     Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
        processed_data_path = Path.home().joinpath('.ludwig_cache') \
            .joinpath("reuters_" + str(self._cur_version)).joinpath('processed.csv')
        result = os.path.isfile(processed_data_path)
        column_names = ["text", "class"]
        if result:
            return pd.read_csv(processed_data_path, names=column_names)
        else:
            raise FileNotFoundError("The transformed data for reuters does not exist so cant return pandas dataframe")

