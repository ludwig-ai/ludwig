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
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class SantanderValuePredictionLoader(DatasetLoader):
    """The Santander Value Prediction Challenge dataset.

    https://www.kaggle.com/c/santander-value-prediction-challenge
    """

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        processed_df = super().transform_dataframe(dataframe)
        # Ensure feature column names are strings (some are numeric); keep special names as is
        processed_df.columns = ["C" + str(col) for col in processed_df.columns]
        processed_df.rename(columns={"CID": "ID", "Ctarget": "target", "Csplit": "split"}, inplace=True)
        return processed_df
