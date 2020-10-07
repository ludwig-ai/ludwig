import csv
import pandas as pd
from typing import Dict
from ludwig.datasets.base_dataset import BaseDataset
import shutil
import os


"""A class to process the ohsumed training data"""


class OhsuMed(BaseDataset):

    def __init__(self):
        initial_path = os.path.abspath(os.path.dirname(__file__))
        self.__source_location =  os.path.join(initial_path, "../text/ohsumed/ohsumed-allcats.csv")
        self.__dest_location = os.path.join(initial_path, "~/.ludwig_cache/ohsumed-allcats.csv")

    """Download the raw data to the ludwig cache in the format ~/.ludwig_cache/id
    where is is represented by the name.version of the dataset
    args:
        self (BaseDataset): A pointer to the current class
        dataset_name: The name of the dataset we need to retrieve
    """
    def download(self, dataset_name) -> None:
        shutil.copy(src, dst)

    """Process the dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API
       args:
           self (BaseDataset): A handle to the current class
           dict_reader (csv.DictReader): a pointer to a dictionary that can be read
       :returns:
           a pandas dataframe
    """
    def process(self, dict_reader: csv.DictReader) -> Dict:
        return {}

    """Now that the data is processed load and return it as a pandas dataframe
     Args:
         self (BaseDataset): A handle to the current class
     Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
       return pd.DataFrame()

