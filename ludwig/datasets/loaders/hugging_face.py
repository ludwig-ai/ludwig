# Copyright (c) 2022 Predibase, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from typing import Dict

import datasets
import pandas as pd

from ludwig.constants import TEST, TRAIN, VALIDATION
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

SPLITS = [TRAIN, VALIDATION, TEST]
logger = logging.getLogger(__name__)


class HFLoader(DatasetLoader):
    """HFLoader differs from all other DatasetLoaders because of how it loads data through the Hugging Face
    datasets API instead of saving any files to the cache.

    The config for HFLoader contains two unique parameters, huggingface_dataset_id and huggingface_subsample, that
    identify which dataset and which subsample of that dataset to load in.
    """

    def load_hf_to_dict(self, hf_id: str, hf_subsample: str) -> Dict[str, pd.DataFrame]:
        """Datasets.load_dataset() returns a mapping of type datasets.dataset_dict.DatasetDict that maps a string
        in.

        ["train", "validation", "test"] to a dataset of type datasets.arrow_dataset.Dataset.

        This function converts each of the datasets in the original mapping to a pandas DataFrame and returns a new
        dictionary that maps a string in ["train", "validation", "test"] to the corresponding DataFrame.
        """
        dataset_dict = datasets.load_dataset(path=hf_id, name=hf_subsample)
        new_dict = {}
        for split in dataset_dict:
            # Convert from HF DatasetDict type to a dictionary of pandas dataframes
            new_dict[split] = dataset_dict[split].to_pandas()
        return new_dict

    def load(self, hf_id, hf_subsample, split=False) -> pd.DataFrame:
        """When load() is called, HFLoader calls the datasets API to return all of the data in a HuggingFace
        DatasetDict, converts it to a dictionary of pandas dataframes, and returns either three dataframes
        containing train, validation, and test data or one dataframe that is the concatenation of all three
        depending on whether `split` is set to True or False.

        Note that some datasets may not provide a validation set or a test set. In this case:
        - If split is True, the DataFrames corresponding to the missing sets are initialized to be empty
        - If split is False, the "split" column in the resulting DataFrame will reflect the fact that there is no
          validation/test split (i.e., there will be no 1s/2s)

        A train set should always be provided by Hugging Face
        """
        self.config.huggingface_dataset_id = hf_id
        self.config.huggingface_subsample = hf_subsample
        dataset_dict = self.load_hf_to_dict(
            hf_id=hf_id,
            hf_subsample=hf_subsample,
        )
        if split:  # For each split, either return the appropriate dataframe or an empty dataframe
            for spl in SPLITS:
                if spl not in dataset_dict:
                    logger.warning(f"No {spl} set found in provided Hugging Face dataset. Skipping {spl} set.")
            train_df = dataset_dict[TRAIN] if TRAIN in dataset_dict else pd.DataFrame()
            validation_df = dataset_dict[VALIDATION] if VALIDATION in dataset_dict else pd.DataFrame()
            test_df = dataset_dict[TEST] if TEST in dataset_dict else pd.DataFrame()

            return train_df, validation_df, test_df
        else:
            dataset_list = []
            for spl in dataset_dict:
                dataset_dict[spl]["split"] = SPLITS.index(spl)  # Add a column containing 0s, 1s, and 2s denoting splits
                dataset_list.append(dataset_dict[spl])
            return pd.concat(dataset_list)
