import csv
import os
import yaml
import shutil
from pathlib import Path
import pandas as pd
from ludwig.datasets.base_dataset import BaseDataset

"""A class to download, process and return the ohsumed training data to be plugged into the Ludwig training API"""


class OhsuMed(BaseDataset):

    def __init__(self):
        self.__initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self.__initial_path, "../text/versions.yaml")
        self._source_location = os.path.join(self.__initial_path, "../text/ohsumed/ohsumed-allcats.csv")
        self._cache_dir = str(Path.home().joinpath('.ludwig_cache'))
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self._cur_version = self._config_file_contents["text"]["ohsumed"]
        self._artifact_dir = str(Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_"+str(self._cur_version)))
        if not os.path.exists(self._artifact_dir):
            os.makedirs(self._artifact_dir)
        self._result_dict={}

    """Download the ohsumed raw data to the ludwig cache in the format ~/.ludwig_cache/id
       where is is represented by the name.version of the dataset
       :param dataset_name: (str) the name of the dataset we need to retrieve.
       Returns: 
          None
    """
    def download(self, dataset_name) -> None:
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_"
                                                       + str(self._cur_version)).joinpath('raw.csv')
        shutil.copy(self._source_location, download_path)
        result = os.path.isfile(download_path)
        if not result:
            raise FileNotFoundError("The raw data was not downloaded correctly")

    """Process the ohsumed dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API, to do this we create
       a new dictionary that contains the KV pairs in the format that we need.
       Returns:
           None
    """
    def process(self) -> None:
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_"
                                                                       + str(self._cur_version)).joinpath('raw.csv')
        processed_data_path = Path.home().joinpath('.ludwig_cache')\
            .joinpath("ohsumed_"+ str(self._cur_version)).joinpath('processed.csv')
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

    """Now that the ohsumed data is processed load and return it as a pandas dataframe
       Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
        processed_data_path = Path.home().joinpath('.ludwig_cache') \
            .joinpath("ohsumed_" + str(self._cur_version)).joinpath('processed.csv')
        result = os.path.isfile(processed_data_path)
        column_names = ["text","class"]
        if result:
            return pd.read_csv(processed_data_path, names=column_names)
        else:
            raise FileNotFoundError("The transformed data for ohsumed does not exist so cant return pandas dataframe")

