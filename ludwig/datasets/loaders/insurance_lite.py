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
import os

import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class InsuranceLiteLoader(DatasetLoader):
    """Health Insurance Cross Sell Prediction Predict Health Insurance Owners' who will be interested in Vehicle
    Insurance https://www.kaggle.com/datasets/arashnic/imbalanced-data-practice."""

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_dataframe(dataframe)
        # Make image paths relative to dataset root directory
        df["image_path"] = df["image_path"].apply(
            lambda x: os.path.join("Fast_Furious_Insured", "trainImages", os.path.basename(x))
        )
        return df
