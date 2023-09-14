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
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class ConsumerComplaintsLoader(DatasetLoader):
    """The Consumer Complaints dataset."""

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        """Loads a file into a dataframe."""

        consumer_complaints_df = pd.read_csv(file_path)
        consumer_complaints_df = preprocess_df(consumer_complaints_df)

        return consumer_complaints_df


def preprocess_df(df):
    """Preprocesses the dataframe.

        - Remove all rows with missing values in the following columns:
            - Consumer complaint narrative
            - Issue
            - Product

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    return df.dropna(subset=["Consumer complaint narrative", "Issue", "Product"])
