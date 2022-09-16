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


class AGNewsLoader(DatasetLoader):
    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        processed_df = super().transform_dataframe(dataframe)
        # Maps class_index to class name.
        class_names = ["", "world", "sports", "business", "sci_tech"]
        # Adds new column 'class' by mapping class indexes to strings.
        processed_df["class"] = processed_df.class_index.apply(lambda i: class_names[i])
        # Agnews has no validation split, only train and test (0, 2). For convenience, we'll designate the first 5% of
        # each class from the training set as the validation set.
        val_set_n = int((len(processed_df) * 0.05) // len(class_names))  # rows from each class in validation set.
        for ci in range(1, 5):
            # For each class, reassign the first val_set_n rows of the training set to validation set.
            train_rows = processed_df[(processed_df.split == 0) & (processed_df.class_index == ci)].index
            processed_df.loc[train_rows[:val_set_n], "split"] = 1
        return processed_df
