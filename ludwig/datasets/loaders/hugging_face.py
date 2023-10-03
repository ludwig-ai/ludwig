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
    def load_hf_to_dict(self, hf_id: str, hf_subsample: str) -> Dict[str, pd.DataFrame]:
        # Convert from HF DatasetDict type to a dictionary of pandas dataframes
        dataset_dict = datasets.load_dataset(path=hf_id, name=hf_subsample)
        new_dict = {}
        for split in dataset_dict:
            new_dict[split] = dataset_dict[split].to_pandas()
        return new_dict

    def load(self, hf_id, hf_subsample, split=False) -> pd.DataFrame:
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
                dataset_dict[spl]["split"] = SPLITS.index(spl)
                dataset_list.append(dataset_dict[spl])
            return pd.concat(dataset_list)
