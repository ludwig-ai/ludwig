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
import datasets
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class HFText2TextGenerationLoader(DatasetLoader):
    def load_hf_to_dict(self, hf_id: str, hf_subset: str) -> dict:
        dataset_dict = datasets.load_dataset(path=hf_id, name=hf_subset)
        new_dict = {}
        for split in dataset_dict:
            new_dict[split] = dataset_dict[split].to_pandas()
        return new_dict

    def load(self, split=False, kaggle_username=None, kaggle_key=None) -> pd.DataFrame:
        dataset_dict = self.load_hf_to_dict(
            hf_id=self.config["hf_id"],
            hf_subset=self.config["hf_subset"],
        )
        if split:
            if "train" in dataset_dict:
                train_df = self.load_hf_to_dataframe(
                    hf_id=self.config["hf_id"],
                    hf_subset=self.config["hf_subset"],
                )["train"]
            else:
                train_df = None
            if "validation" in dataset_dict:
                validation_df = self.load_hf_to_dataframe(
                    hf_id=self.config["hf_id"],
                    hf_subset=self.config["hf_subset"],
                )["validation"]
            else:
                validation_df = None
            if "test" in dataset_dict:
                test_df = self.load_hf_to_dataframe(
                    hf_id=self.config["hf_id"],
                    hf_subset=self.config["hf_subset"],
                )["test"]
            else:
                test_df = None
            return train_df, validation_df, test_df
        else:
            dataset_list = [dataset_dict[split] for split in dataset_dict]
            return pd.concat(dataset_list)
