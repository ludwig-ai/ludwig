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
import numpy as np
import pandas as pd

from ludwig.constants import SPLIT
from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class RandomSplitLoader(DatasetLoader):
    """Adds a random split column to the dataset, with fixed proportions of:
     train: 70%
     validation: 10%
     test: 20%
    ."""

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_dataframe(dataframe)
        df[SPLIT] = np.random.choice(3, len(df), p=(0.7, 0.1, 0.2)).astype(np.int8)
        return df
