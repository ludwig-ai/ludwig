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


class CreditCardFraudLoader(DatasetLoader):
    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        processed_df = super().transform_dataframe(dataframe)
        # Train/Test split like https://www.kaggle.com/competitions/1056lab-fraud-detection-in-credit-card/overview
        processed_df = processed_df.sort_values(by=["Time"])
        processed_df.loc[:198365, "split"] = 0
        processed_df.loc[198365:, "split"] = 2
        processed_df.split = processed_df.split.astype(int)
        return processed_df
