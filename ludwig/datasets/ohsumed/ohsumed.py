import csv
import pandas as pd
from typing import Dict
from ludwig.datasets.base_dataset import BaseDataset
import shutil
import os


"""A class to process the ohsumed training data"""
import yaml


class OhsuMed(BaseDataset):

    def __init__(self):
        self.__initial_path = os.path.abspath(os.path.dirname(__file__))
        self.__source_location =  os.path.join(self.__initial_path , "../text/ohsumed/ohsumed-allcats.csv")
        self.__cache_location = os.path.join(self.__initial_path, "../../../.ludwig_cache/")
        self.__config_file_path = os.path.join(self.__initial_path , "../text/versions.yaml")
        with open(self.__config_file_path) as config_file:
            self.__config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        cur_version = self.__config_file_contents["text"]["ohsumed"]
        self.__dest_location = self.__cache_location + "ohsumed_" + str(cur_version) + ".csv"
        self.__result_dict={}

    """Download the raw data to the ludwig cache in the format ~/.ludwig_cache/id
    where is is represented by the name.version of the dataset
    args:
        self (BaseDataset): A pointer to the current class
        dataset_name: The name of the dataset we need to retrieve
    return:
        true or false depending on whether the file exists in the new location
    """
    def download(self, dataset_name) -> bool:
        shutil.copy(self.__source_location, self.__dest_location)
        return os.path.isfile(self.__dest_location)


    """Process the dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API
       args:
           self (BaseDataset): A handle to the current class
           dict_reader (csv.DictReader): a pointer to a dictionary that can be read
       :returns:
           a dictionary containing KV pairs in the format of the training data
    """
    def process(self) -> Dict:
        result = os.path.isfile(self.__dest_location)
        if result != True:
            self.download("ohsumed")
        dict_reader = csv.DictReader(open(self.__dest_location))
        value_to_store = None
        for row in dict_reader:
            for key, value in row.items():
                if key == "class":
                    value_to_store = value
                else:
                    key_to_store = value
                    self.__result_dict[key_to_store] = value_to_store
        return self.__result_dict

    """Now that the data is processed load and return it as a pandas dataframe
     Args:
         self (BaseDataset): A handle to the current class
     Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
       result = os.path.isfile(self.__dest_location)
       if result != True:
            self.download("ohsumed")
       processed_result = self.process()
       return pd.DataFrame(list(processed_result.items()), columns=['text', 'class'])

