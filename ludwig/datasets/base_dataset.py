import abc
import pandas as pd


"""An abstract base class that defines a set of methods for download,
preprocess and plug data into the ludwig training API"""


class BaseDataset(metaclass=abc.ABCMeta):

    """Download the raw data to the ludwig cache in the format ~/.ludwig_cache/id
       where is is represented by the name.version of the dataset
       :param dataset_name: (str) the name of the dataset we need to retrieve.
       Returns:
          None
    """
    @abc.abstractmethod
    def download(self, dataset_name) -> None:
        raise NotImplementedError("You will need to implement the download method to download the training data")

    """Process the dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API
       Returns:
           None
    """
    @abc.abstractmethod
    def process(self) -> None:
        raise NotImplementedError("You will need to implement the method to process the training data")

    """Now that the data is processed load and return it as a pandas dataframe
       Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
       raise NotImplementedError("You will need to implement the method to return the processed data as a pandas dataframe")

