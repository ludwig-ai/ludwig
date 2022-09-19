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
from typing import Optional

import pandas as pd

from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class HiggsLoader(DatasetLoader):
    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, add_validation_set=True):
        super().__init__(config, cache_dir)
        self.add_validation_set = add_validation_set

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        """Loads a file into a dataframe."""
        return pd.read_csv(file_path, header=None)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        processed_df = super().transform_dataframe(dataframe)
        if self.add_validation_set:
            processed_df["split"] = [0] * 10000000 + [1] * 500000 + [2] * 500000
        else:
            processed_df["split"] = [0] * 10500000 + [2] * 500000
        return processed_df
