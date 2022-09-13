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
from typing import List, Optional

import pandas as pd

from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class EmailLoader(DatasetLoader):
    """Converts a folder of EML files to parquet dataset.

    If folders are present in the input directory, they are assumed to represent different classes and an additional
    column is added to the exported dataset using the directory names as labels.
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, add_binary_columns: bool = True):
        super().__init__(config, cache_dir=cache_dir)
        self.add_binary_columns = add_binary_columns

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        """Load dataset files into a dataframe."""

        return None
