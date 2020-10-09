import abc
from typing import Dict
import pandas as pd


"""An abstract base class that defines a set of methods for download,
preprocess and plug data into the ludwig training API"""


class BaseDataset(metaclass=abc.ABCMeta):

    """Download the raw data to the ludwig cache in the format ~/.ludwig_cache/id
    where is is represented by the name.version of the dataset
    args:
        self (BaseDataset): A pointer to the current class
        dataset_name: The name of the dataset we need to retrieve
    """
    @abc.abstractmethod
    def download(self, dataset_name) -> None:
        raise NotImplementedError("You will need to implement the download method to download the training data")

    """Process the dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API
       args:
           self (BaseDataset): A handle to the current class
           dict_reader (csv.DictReader): a pointer to a dictionary that can be read
       :returns:
           a pandas dataframe
    """
    @abc.abstractmethod
    def process(self) -> Dict:
        raise NotImplementedError("You will need to implement the method to process the training data")

    """Now that the data is processed load and return it as a pandas dataframe
     Args:
         self (BaseDataset): A handle to the current class
     Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
       raise NotImplementedError("You will need to implement the method to return the processed data as a pandas dataframe")

