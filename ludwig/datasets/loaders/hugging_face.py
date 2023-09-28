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

import datasets
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
splits = [TRAIN, VALIDATION, TEST]


class HFLoader(DatasetLoader):
    def load_hf_to_dict(self, hf_id: str, hf_subsample: str) -> dict:
        # Convert from HF DatasetDict type to dict
        dataset_dict = datasets.load_dataset(path=hf_id, name=hf_subsample)
        new_dict = {}
        for split in dataset_dict:
            new_dict[split] = dataset_dict[split].to_pandas()
        return new_dict

    def load(self, split=False, kaggle_username=None, kaggle_key=None) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        dataset_dict = self.load_hf_to_dict(
            hf_id=self.config.huggingface_dataset_id,
            hf_subsample=self.config.huggingface_subsample,
        )
        if split:  # For each split, either return the appropriate dataframe or an empty dataframe
            if TRAIN in dataset_dict:
                train_df = dataset_dict[TRAIN]
            else:
                logger.warning("No training set found in provided Hugging Face dataset. Skipping training set.")
                train_df = pd.DataFrame()
            if VALIDATION in dataset_dict:
                validation_df = dataset_dict[VALIDATION]
            else:
                logger.warning("No validation set found in provided Hugging Face dataset. Skipping validation set.")
                validation_df = pd.DataFrame()
            if TEST in dataset_dict:
                test_df = dataset_dict[TEST]
            else:
                logger.warning("No test set found in provided Hugging Face dataset. Skipping test set.")
                test_df = pd.DataFrame()

            return train_df, validation_df, test_df
        else:
            dataset_list = []
            for split in dataset_dict:
                dataset_dict[split]["split"] = splits.index(split)
                dataset_list.append(dataset_dict[split])
            return pd.concat(dataset_list)
